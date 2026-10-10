from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None


st.set_page_config(
    page_title="Predictive Maintenance | AI4I 2020",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET_DEFAULT = "Machine failure"
KNOWN_ID_COLUMNS = {"UDI", "Product ID"}
KNOWN_LEAKAGE_COLUMNS = {"TWF", "HDF", "PWF", "OSF", "RNF"}
RANDOM_STATE_DEFAULT = 42

# FastAPI integration. Override the URL with the environment variable if the API
# is hosted elsewhere. By default, both services run on this computer.
PROJECT_ROOT = Path(__file__).resolve().parent
API_BASE_URL = os.getenv("PREDICTIVE_MAINTENANCE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_MODEL_DIR = PROJECT_ROOT / "models" / "api_serving"
AI4I_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ucimlrepo may expose the AI4I columns without their units, while the CSV
# uses the unit-bearing names. Canonicalize both sources before feature
# selection, training, and API export so the same schema is always used.
AI4I_COLUMN_ALIASES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}


def normalize_ai4i_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize alternate AI4I column names to the canonical schema."""
    result = data.copy()
    existing = set(map(str, result.columns))
    rename_map = {
        alias: canonical
        for alias, canonical in AI4I_COLUMN_ALIASES.items()
        if alias in existing and canonical not in existing
    }
    if rename_map:
        result = result.rename(columns=rename_map)
    return result


API_PUBLISH_LOG = API_MODEL_DIR / "publish_debug.log"


def _write_api_publish_log(event: str, **details: Any) -> None:
    """Append diagnostics for the Streamlit-to-FastAPI model publication flow."""
    try:
        API_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **details,
        }
        with API_PUBLISH_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Logging must never prevent training or publishing.
        pass


@st.cache_data(show_spinner=False)
def load_ai4i_from_uci() -> pd.DataFrame:
    """Download the UCI AI4I 2020 Predictive Maintenance dataset."""
    if fetch_ucirepo is None:
        raise ImportError(
            "ucimlrepo is not installed. Install it with: pip install ucimlrepo"
        )
    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features
    targets = dataset.data.targets
    data = pd.concat([features, targets], axis=1)
    # Handle the case where the target may already be included in features.
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def dataframe_signature(data: pd.DataFrame) -> str:
    """Return a stable signature for the loaded dataset."""
    h = hashlib.md5()
    h.update("|".join(map(str, data.columns)).encode("utf-8"))
    h.update("|".join(map(str, data.dtypes)).encode("utf-8"))
    try:
        h.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    except Exception:
        h.update(data.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def safe_default_features(data: pd.DataFrame, target: str) -> list[str]:
    candidates = [
        c for c in data.columns
        if c != target and c not in KNOWN_ID_COLUMNS and c not in KNOWN_LEAKAGE_COLUMNS
    ]
    # Also exclude high-cardinality text fields likely to be IDs.
    safe = []
    n = max(len(data), 1)
    for col in candidates:
        if data[col].dtype == "object" or pd.api.types.is_string_dtype(data[col]):
            unique_ratio = data[col].nunique(dropna=True) / n
            if unique_ratio > 0.50:
                continue
        safe.append(col)
    return safe


def default_positive_label(values: list[Any]) -> Any:
    for value in values:
        try:
            if value == 1 or str(value).strip() == "1":
                return value
        except Exception:
            pass
    return values[-1]


def positive_label_index(values: list[Any], chosen: Any) -> int:
    for i, value in enumerate(values):
        try:
            if value == chosen:
                return i
        except Exception:
            continue
    return len(values) - 1


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build dtype-safe preprocessing for numerical and categorical columns.

    Numeric columns receive median imputation and optional scaling.
    Every non-numeric column (including pandas StringDtype columns) is treated
    as categorical and receives most-frequent imputation plus one-hot encoding.
    """
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    transformers = []

    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were selected.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def create_model(model_name: str, seed: int = RANDOM_STATE_DEFAULT):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=seed), True
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed), False
    if model_name == "KNN":
        return KNeighborsClassifier(), True
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=1
        ), False
    if model_name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        ), False
    raise ValueError(f"Unsupported model: {model_name}")


def create_pipeline(model_name: str, X_train: pd.DataFrame, sampler, seed: int):
    estimator, scale_numeric = create_model(model_name, seed)
    return ImbPipeline(steps=[
        ("preprocessing", build_preprocessor(X_train, scale_numeric)),
        ("sampler", sampler),
        ("model", estimator),
    ])


def parameter_space(model_name: str) -> dict[str, list[Any]]:
    spaces = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "Decision Tree": {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 8, 16, 24],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.8],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.03, 0.1, 0.2],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [1, 5],
        },
    }
    return spaces[model_name]


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def run_training(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    positive_label: Any,
    test_size: float,
    selected_models: list[str],
    sampler_mode: str,
    do_tuning: bool,
    cv_splits: int,
    n_iter: int,
    seed: int,
):
    modeling = data[features + [target]].dropna(subset=[target]).copy()
    X = modeling[features].copy()
    y_raw = modeling[target]
    y = (y_raw == positive_label).astype(int)

    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < cv_splits:
        raise ValueError(
            f"Need both target classes to have at least {cv_splits} observations. "
            f"Class counts after mapping: {class_counts.to_dict()}"
        )
    if y.nunique() != 2:
        raise ValueError("Selected target and positive label did not produce two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    if sampler_mode == "None":
        sampler = "passthrough"
    elif sampler_mode == "Moderate oversampling (minority/majority = 0.5)":
        sampler = RandomOverSampler(sampling_strategy=0.5, random_state=seed)
    else:
        sampler = RandomOverSampler(sampling_strategy=1.0, random_state=seed)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    trained_models = {}
    probabilities_by_model = {}
    predictions_by_model = {}
    result_rows = []
    search_summaries = {}

    for model_name in selected_models:
        base_pipeline = create_pipeline(model_name, X_train, sampler, seed)
        if do_tuning:
            search = RandomizedSearchCV(
                estimator=base_pipeline,
                param_distributions=parameter_space(model_name),
                n_iter=n_iter,
                scoring=SCORING,
                refit="pr_auc",
                cv=cv,
                random_state=seed,
                n_jobs=-1,
                verbose=0,
                error_score="raise",
                return_train_score=False,
                pre_dispatch="2*n_jobs",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            cv_row = {
                "CV Average Precision": float(search.best_score_),
                "CV ROC-AUC": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
                "CV Recall": float(search.cv_results_["mean_test_recall"][search.best_index_]),
                "CV F1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
                "Best parameters": json.dumps(search.best_params_, default=str),
            }
            search_summaries[model_name] = {
                "best_params": search.best_params_,
                "best_cv_ap": float(search.best_score_),
                "cv_results": pd.DataFrame(search.cv_results_),
            }
        else:
            cv_scores = cross_validate(
                base_pipeline,
                X_train,
                y_train,
                scoring=SCORING,
                cv=cv,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )
            fitted = clone(base_pipeline)
            fitted.fit(X_train, y_train)
            cv_row = {
                "CV Average Precision": float(np.mean(cv_scores["test_pr_auc"])),
                "CV ROC-AUC": float(np.mean(cv_scores["test_roc_auc"])),
                "CV Recall": float(np.mean(cv_scores["test_recall"])),
                "CV F1": float(np.mean(cv_scores["test_f1"])),
                "Best parameters": "Baseline configuration",
            }
            search_summaries[model_name] = {
                "best_params": {},
                "best_cv_ap": cv_row["CV Average Precision"],
                "cv_results": pd.DataFrame(cv_scores),
            }

        probabilities = fitted.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, probabilities)

        row = {
            "Model": model_name,
            **cv_row,
            "Threshold": 0.5,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "PR-AUC (trapezoidal)": auc(pr_recall, pr_precision),
            "Average Precision": average_precision_score(y_test, probabilities),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }
        trained_models[model_name] = fitted
        probabilities_by_model[model_name] = probabilities
        predictions_by_model[model_name] = predictions
        result_rows.append(row)

    results = pd.DataFrame(result_rows).sort_values(
        "CV Average Precision", ascending=False
    ).reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "target": target,
        "positive_label": positive_label,
        "trained_models": trained_models,
        "probabilities": probabilities_by_model,
        "predictions": predictions_by_model,
        "results": results,
        "search_summaries": search_summaries,
        "best_model_name": results.iloc[0]["Model"],
        "split_time": time.time(),
        "sampler_mode": sampler_mode,
        "do_tuning": do_tuning,
        "cv_splits": cv_splits,
        "test_size": test_size,
        "seed": seed,
    }


def make_confusion_figure(y_true, probabilities, threshold: float, title: str):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: no failure", "Predicted: failure"],
        y=["Actual: no failure", "Actual: failure"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Predicted class", yaxis_title="Actual class")
    return fig, cm, predictions


# ------------------------------
# FastAPI model registry utilities
# ------------------------------
def _safe_model_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError("Could not derive a safe file name from the model name.")
    return slug


def export_ai4i_models_for_api(trained: dict[str, Any]) -> tuple[bool, str]:
    """Persist trained AI4I pipelines for the separate FastAPI process.

    The function writes a persistent log first, including explicit reasons for
    validation skips. This makes failures observable even after Streamlit reruns.
    """
    features = list(trained.get("features", []))
    target = str(trained.get("target", ""))
    positive_label = trained.get("positive_label")
    model_names = list(trained.get("trained_models", {}).keys())

    _write_api_publish_log(
        "export_called",
        project_root=str(PROJECT_ROOT),
        api_model_dir=str(API_MODEL_DIR),
        target=target,
        positive_label=repr(positive_label),
        positive_label_type=type(positive_label).__name__,
        features=features,
        fitted_models=model_names,
    )

    def skip(reason: str) -> tuple[bool, str]:
        _write_api_publish_log("export_skipped", reason=reason)
        return False, reason

    if target != "Machine failure":
        return skip(f"API export skipped: target is {target!r}; required 'Machine failure'.")
    if str(positive_label).strip() not in {"1", "1.0"}:
        return skip(
            f"API export skipped: positive class is {positive_label!r} "
            "(type " + type(positive_label).__name__ + "); required 1."
        )
    if len(features) != len(AI4I_FEATURES) or set(features) != set(AI4I_FEATURES):
        return skip(
            "API export skipped: feature schema mismatch. "
            f"Actual features={features!r}; expected exactly={AI4I_FEATURES!r}."
        )

    API_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = trained.get("results", pd.DataFrame())
    models = trained.get("trained_models", {})
    if not models:
        return skip("API export skipped: there are no fitted pipelines to publish.")

    model_entries = []
    new_artifact_names = set()
    try:
        for name, pipeline in models.items():
            slug = _safe_model_slug(name)
            artifact_name = f"{slug}.joblib"
            artifact_path = API_MODEL_DIR / artifact_name
            tmp_path = API_MODEL_DIR / f".{slug}.tmp.joblib"
            _write_api_publish_log(
                "writing_model_artifact",
                model_name=name,
                temporary_path=str(tmp_path),
                destination_path=str(artifact_path),
                pipeline_type=type(pipeline).__name__,
            )
            joblib.dump(pipeline, tmp_path)
            os.replace(tmp_path, artifact_path)
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                raise OSError(f"Model artifact was not created or is empty: {artifact_path}")
            new_artifact_names.add(artifact_name)

            cv_ap = None
            if not results.empty and "Model" in results.columns and "CV Average Precision" in results.columns:
                match = results.loc[results["Model"] == name, "CV Average Precision"]
                if not match.empty:
                    try:
                        cv_ap = float(match.iloc[0])
                    except (TypeError, ValueError):
                        cv_ap = None

            model_entries.append({
                "name": name,
                "artifact": artifact_name,
                "cross_validated_average_precision": cv_ap,
            })

        best_model_name = str(trained.get("best_model_name", model_entries[0]["name"]))
        catalog = {
            "project": "AI4I 2020 Predictive Maintenance",
            "target": "Machine failure",
            "positive_class": 1,
            "features": features,
            "default_threshold": 0.50,
            "default_model": best_model_name,
            "models": model_entries,
            "training_configuration": {
                "sampler_mode": trained.get("sampler_mode"),
                "hyperparameter_tuning": bool(trained.get("do_tuning", False)),
                "cross_validation_folds": trained.get("cv_splits"),
                "test_size": trained.get("test_size"),
                "random_seed": trained.get("seed"),
            },
            "runtime_versions": {
                "python": os.sys.version.split()[0],
                "scikit_learn": __import__("sklearn").__version__,
                "joblib": joblib.__version__,
            },
        }
        catalog_path = API_MODEL_DIR / "catalog.json"
        tmp_catalog = API_MODEL_DIR / ".catalog.tmp.json"
        tmp_catalog.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        os.replace(tmp_catalog, catalog_path)
        if not catalog_path.is_file():
            raise OSError(f"Catalog was not created: {catalog_path}")

        # Remove obsolete model files only after the new catalog has been committed.
        for old_path in API_MODEL_DIR.glob("*.joblib"):
            if old_path.name not in new_artifact_names:
                try:
                    old_path.unlink()
                except OSError:
                    pass

        message = (
            f"Published {len(model_entries)} AI4I model(s) to the FastAPI registry. "
            f"Default model: {best_model_name}. Registry: {API_MODEL_DIR}"
        )
        _write_api_publish_log(
            "export_succeeded",
            catalog_path=str(catalog_path),
            artifact_names=sorted(new_artifact_names),
            model_names=[item["name"] for item in model_entries],
        )
        return True, message
    except Exception as exc:
        _write_api_publish_log(
            "export_exception",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        raise



# ------------------------------
# Sidebar / dataset loading
# ------------------------------
st.title("Predictive Maintenance Classification")
st.caption("AI4I 2020 · Data Mining · Interactive supervised-learning dashboard")

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose dataset source",
        ["AI4I 2020 (UCI, automatic download)", "Upload a CSV file"],
        index=0,
    )
    uploaded_file = None
    if source == "Upload a CSV file":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.caption("The XAI section is reserved for a later project phase.")

raw_data = None
load_error = None
if source == "AI4I 2020 (UCI, automatic download)":
    try:
        with st.spinner("Downloading AI4I 2020 dataset from UCI..."):
            raw_data = load_ai4i_from_uci()
    except Exception as exc:
        load_error = str(exc)
else:
    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
        except Exception as exc:
            load_error = f"Could not read the uploaded CSV: {exc}"

if raw_data is None:
    st.info(
        "Load the official UCI dataset automatically or upload a CSV to begin. "
        "If UCI download fails in the deployment environment, upload the AI4I CSV instead."
    )
    if load_error:
        st.error(f"Dataset loading failed: {load_error}")
    st.stop()

if raw_data.empty:
    st.error("The loaded dataset is empty.")
    st.stop()

# Clean duplicate columns, then standardize UCI/CSV variations in AI4I names.
# This must happen before target/feature detection, otherwise the app sees
# "Air temperature" instead of "Air temperature [K]" and refuses API export.
raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()].copy()
raw_data = normalize_ai4i_column_names(raw_data)
data_sig = dataframe_signature(raw_data)

# ------------------------------
# Main navigation
# ------------------------------
tab_data, tab_train, tab_eval, tab_api_predict, tab_about = st.tabs([
    "1 · Data & EDA",
    "2 · Preparation & Training",
    "3 · Evaluation",
    "4 · Machine Prediction (FastAPI)",
    "5 · About",
])

with tab_data:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw_data):,}")
    c2.metric("Columns", f"{raw_data.shape[1]:,}")
    c3.metric("Missing cells", f"{int(raw_data.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(raw_data.duplicated().sum()):,}")

    with st.expander("Preview data", expanded=True):
        st.dataframe(raw_data.head(20), use_container_width=True)

    st.markdown("#### Column types and missing values")
    schema = pd.DataFrame({
        "Column": raw_data.columns,
        "Data type": [str(raw_data[c].dtype) for c in raw_data.columns],
        "Missing values": [int(raw_data[c].isna().sum()) for c in raw_data.columns],
        "Unique values": [int(raw_data[c].nunique(dropna=True)) for c in raw_data.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(raw_data.describe(include="all").T, use_container_width=True)

    st.markdown("#### Exploratory visualizations")
    candidate_targets = list(raw_data.columns)
    default_target_index = candidate_targets.index(TARGET_DEFAULT) if TARGET_DEFAULT in candidate_targets else 0
    eda_target = st.selectbox(
        "Target for class distribution / failure-rate plots",
        candidate_targets,
        index=default_target_index,
        key="eda_target_select",
    )
    plot_frame = raw_data.dropna(subset=[eda_target]).copy()
    target_values = plot_frame[eda_target].dropna().unique().tolist()
    if len(target_values) == 2:
        target_value = st.selectbox(
            "Positive class",
            target_values,
            index=positive_label_index(target_values, default_positive_label(target_values)),
            format_func=lambda x: str(x),
            key="eda_positive_label",
        )
        counts = plot_frame[eda_target].value_counts(dropna=False).reset_index()
        counts.columns = ["Class", "Count"]
        counts["Class"] = counts["Class"].astype(str)
        fig = px.bar(counts, x="Class", y="Count", text="Count", title=f"Class distribution: {eda_target}")
        st.plotly_chart(fig, use_container_width=True)
        counts_pos = plot_frame.assign(_positive=(plot_frame[eda_target] == target_value).astype(int))
    else:
        counts_pos = None
        st.warning("This target has other than two distinct values. Select a binary target for failure-rate analysis.")

    numeric_columns = raw_data.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        dist_feature = st.selectbox(
            "Numerical feature distribution",
            numeric_columns,
            key="eda_dist_feature",
        )
        fig = px.histogram(
            raw_data,
            x=dist_feature,
            marginal="box",
            nbins=40,
            title=f"Distribution of {dist_feature}",
        )
        st.plotly_chart(fig, use_container_width=True)

        corr = raw_data[numeric_columns].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Pearson correlation heatmap (numeric columns)",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    if counts_pos is not None:
        categorical_candidates = [
            c for c in raw_data.columns
            if c != eda_target and (
                raw_data[c].dtype == "object"
                or isinstance(raw_data[c].dtype, pd.CategoricalDtype)
            )
            and raw_data[c].nunique(dropna=True) <= 20
        ]
        if categorical_candidates:
            cat_feature = st.selectbox(
                "Categorical feature for failure-rate comparison",
                categorical_candidates,
                key="eda_cat_feature",
            )
            rate = (
                counts_pos.groupby(cat_feature, dropna=False)["_positive"]
                .agg(Total="size", Failures="sum", Failure_Rate="mean")
                .reset_index()
            )
            rate["Failure rate (%)"] = 100 * rate["Failure_Rate"]
            fig = px.bar(
                rate,
                x=cat_feature,
                y="Failure rate (%)",
                text=rate["Failure rate (%)"].map(lambda v: f"{v:.2f}%"),
                hover_data=["Total", "Failures"],
                title=f"Positive-class rate by {cat_feature}",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_train:
    st.subheader("Prepare data and train classifiers")
    st.write(
        "This AI4I project uses a fixed target: Machine failure, where class 1 means a machine failure. "
        "Choose the split and model settings, then train with cross-validation on the training portion only. "
        "The test set is reserved for evaluation."
    )

    # The project target and positive class are fixed. No target/positive-class
    # selectors are shown in this tab. Other CSVs can be used only if they have
    # the same target; AI4I-compatible data uses the six approved predictors.
    target_col = TARGET_DEFAULT
    if target_col not in raw_data.columns:
        st.error(
            "Training is configured for the AI4I predictive-maintenance task. "
            "Load the AI4I 2020 dataset or a CSV containing the 'Machine failure' target. "
            "Target selection is intentionally fixed for this project."
        )
        st.stop()

    target_values = raw_data[target_col].dropna().unique().tolist()
    try:
        numeric_target_values = {float(value) for value in target_values}
    except (TypeError, ValueError):
        numeric_target_values = set()

    if numeric_target_values != {0.0, 1.0}:
        st.error(
            "The fixed target 'Machine failure' must contain exactly classes 0 and 1, "
            "where 1 represents a machine failure."
        )
        st.stop()

    # Preserve the raw label type so comparisons work for integer, float, or
    # string-encoded CSV target values. AI4I normally uses integer labels.
    positive_candidates = [
        value for value in target_values
        if str(value).strip() in {"1", "1.0"}
    ]
    if not positive_candidates:
        st.error("Could not identify class 1 in the 'Machine failure' target.")
        st.stop()
    positive_label = positive_candidates[0]

    # Recognize both the complete UCI/CSV schema and a cleaned AI4I CSV that
    # retains only the six predictors and the target.
    ai4i_core_columns = set(AI4I_FEATURES) | {TARGET_DEFAULT}
    is_ai4i_dataset = ai4i_core_columns.issubset(set(raw_data.columns))

    if is_ai4i_dataset:
        features = AI4I_FEATURES.copy()
        st.caption(
            "Fixed configuration · Target: Machine failure · Positive class: 1 · "
            "Six approved operating-condition predictors"
        )
        st.info(
            "AI4I data detected. Identifiers (UDI, Product ID) and failure-mode columns "
            "(TWF, HDF, PWF, OSF, RNF) are excluded to prevent target leakage."
        )
    else:
        # For other CSVs, keep feature selection available, but the target and
        # positive class remain fixed to the AI4I project definition.
        feature_options = [
            column for column in raw_data.columns
            if column != target_col
            and column not in KNOWN_ID_COLUMNS
            and column not in KNOWN_LEAKAGE_COLUMNS
        ]
        feature_defaults = safe_default_features(raw_data, target_col)
        feature_defaults = [c for c in feature_defaults if c in feature_options]
        if not feature_options or not feature_defaults:
            st.error(
                "No usable predictors remain after excluding the target, identifiers, "
                "and known failure-mode leakage columns."
            )
            st.stop()

        st.caption(
            "Fixed target: Machine failure · Fixed positive class: 1. "
            "Predictor features can be selected for this CSV."
        )
        features = st.multiselect(
            "Predictor features",
            feature_options,
            default=feature_defaults,
            help=(
                "The target and positive class are fixed for this project. "
                "Identifiers and known failure-mode indicators are excluded from predictors."
            ),
            key="model_features_select",
        )
        if not features:
            st.error("Select at least one predictor feature.")
            st.stop()

    config_col1, config_col2, config_col3 = st.columns(3)
    with config_col1:
        test_size = st.slider(
            "Test-set proportion",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="test_size_slider",
        )
    with config_col2:
        cv_splits = st.selectbox(
            "Cross-validation folds",
            [3, 5, 7],
            index=1,
            key="cv_splits_select",
        )
    with config_col3:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            key="random_seed_input",
        )

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest",
        "XGBoost",
    ]
    selected_models = st.multiselect(
        "Models to train",
        model_names,
        default=model_names,
        key="selected_models",
    )

    train_col1, train_col2 = st.columns(2)
    with train_col1:
        sampler_mode = st.selectbox(
            "Class-imbalance strategy",
            [
                "None",
                "Moderate oversampling (minority/majority = 0.5)",
                "Full oversampling (minority/majority = 1.0)",
            ],
            index=0,
            help="Oversampling is applied inside each training fold only, never to the validation or test set.",
            key="sampler_mode_select",
        )
    with train_col2:
        do_tuning = st.checkbox(
            "Optimize hyperparameters with RandomizedSearchCV",
            value=False,
            key="do_tuning_checkbox",
        )

    n_iter = 8
    if do_tuning:
        n_iter = st.slider(
            "Randomized search iterations per model",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            key="n_iter_slider",
        )
        st.caption(
            f"Estimated search load: up to {len(selected_models) * n_iter * cv_splits} CV fits, "
            "plus final model refits. Start with fewer models or iterations if needed."
        )

    st.warning(
        "Known leakage columns (TWF, HDF, PWF, OSF, RNF) and identifiers (UDI, Product ID) "
        "are excluded from the default feature selection. Verify the feature list before training."
    )

    current_config = {
        "dataset_signature": data_sig,
        "target": target_col,
        "features": features,
        "positive_label": str(positive_label),
        "test_size": test_size,
        "cv_splits": cv_splits,
        "seed": int(seed),
        "selected_models": selected_models,
        "sampler_mode": sampler_mode,
        "do_tuning": do_tuning,
        "n_iter": n_iter,
    }
    current_config_signature = hashlib.md5(
        json.dumps(current_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    run_button = st.button(
        "Train selected models",
        type="primary",
        disabled=not selected_models,
        use_container_width=True,
    )

    if run_button:
        try:
            with st.spinner("Training models and running cross-validation..."):
                trained = run_training(
                    data=raw_data,
                    target=target_col,
                    features=features,
                    positive_label=positive_label,
                    test_size=float(test_size),
                    selected_models=selected_models,
                    sampler_mode=sampler_mode,
                    do_tuning=do_tuning,
                    cv_splits=int(cv_splits),
                    n_iter=int(n_iter),
                    seed=int(seed),
                )
            trained["config_signature"] = current_config_signature
            trained["config"] = current_config
            st.session_state["trained_experiment"] = trained
            st.success("Training completed. Open the Evaluation tab to review the results.")

            _write_api_publish_log(
                "training_finished_before_export",
                model_names=list(trained.get("trained_models", {}).keys()),
                target=trained.get("target"),
                positive_label=repr(trained.get("positive_label")),
                features=trained.get("features", []),
            )
            api_export_ok, api_export_message = export_ai4i_models_for_api(trained)
            if api_export_ok:
                st.success(api_export_message)
            else:
                st.info(api_export_message)

            st.dataframe(
                trained["results"][
                    ["Model", "CV Average Precision", "CV ROC-AUC", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]
                ].round(4),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                f"Leading model by cross-validated Average Precision: {trained['best_model_name']}. "
                "The final decision should also consider recall, false alarms, and operational costs."
            )
        except Exception as exc:
            _write_api_publish_log(
                "training_or_auto_publish_exception",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
            st.exception(exc)

with tab_eval:
    st.subheader("Model evaluation and comparison")
    trained = st.session_state.get("trained_experiment")

    if trained is None:
        st.info("Train at least one model in the Preparation & Training tab to populate evaluation results.")
    else:
        if trained.get("config_signature") != current_config_signature if 'current_config_signature' in locals() else False:
            st.warning(
                "The current training settings differ from the settings used for the stored results. "
                "Retrain in the Preparation & Training tab to update the experiment."
            )

        results = trained["results"].copy()
        st.caption(
            "All test metrics below use the same held-out test split. Model choice is ordered by "
            "cross-validated Average Precision, not by selecting the highest test score."
        )
        shown_cols = [
            "Model", "CV Average Precision", "Accuracy", "Precision", "Recall",
            "F1-score", "ROC-AUC", "PR-AUC (trapezoidal)", "Average Precision",
            "TN", "FP", "FN", "TP"
        ]
        st.dataframe(
            results[shown_cols].round(4),
            use_container_width=True,
            hide_index=True,
        )

        metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]
        melted = results[["Model"] + metric_cols].melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            melted,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Test-set metric comparison",
            range_y=[0, 1.05],
        )
        st.plotly_chart(fig, use_container_width=True)

        best_model_name = trained["best_model_name"]
        selected_eval_model = st.selectbox(
            "Model to inspect in detail",
            list(trained["trained_models"].keys()),
            index=list(trained["trained_models"].keys()).index(best_model_name),
            key="detailed_eval_model",
        )
        y_test = trained["y_test"]
        probabilities_by_model = trained["probabilities"]
        best_probs = probabilities_by_model[selected_eval_model]
        selected_row = results.loc[results["Model"] == selected_eval_model].iloc[0]

        threshold = st.slider(
            "Decision threshold for detailed confusion matrix",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            key="eval_threshold_slider",
        )
        conf_fig, cm, selected_predictions = make_confusion_figure(
            y_test,
            best_probs,
            threshold,
            f"Confusion matrix — {selected_eval_model} (threshold {threshold:.2f})",
        )

        tn, fp, fn, tp = cm.ravel()
        detailed_metrics = {
            "Accuracy": accuracy_score(y_test, selected_predictions),
            "Precision": precision_score(y_test, selected_predictions, zero_division=0),
            "Recall": recall_score(y_test, selected_predictions, zero_division=0),
            "F1-score": f1_score(y_test, selected_predictions, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "Balanced accuracy": balanced_accuracy_score(y_test, selected_predictions),
            "ROC-AUC": roc_auc_score(y_test, best_probs),
            "Average Precision": average_precision_score(y_test, best_probs),
        }

        kpi_cols = st.columns(4)
        for col, (metric, value) in zip(kpi_cols, list(detailed_metrics.items())[:4]):
            col.metric(metric, f"{value:.4f}")
        more_cols = st.columns(4)
        for col, (metric, value) in zip(more_cols, list(detailed_metrics.items())[4:]):
            col.metric(metric, f"{value:.4f}")

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(conf_fig, use_container_width=True)
        with right:
            report = classification_report(
                y_test,
                selected_predictions,
                labels=[0, 1],
                target_names=["Negative class (0)", "Positive class (1)"],
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T
            st.markdown("**Classification report at the selected threshold**")
            st.dataframe(report_df.round(4), use_container_width=True)

        st.markdown("#### ROC curves")
        roc_fig = go.Figure()
        for name, probabilities in probabilities_by_model.items():
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            score = roc_auc_score(y_test, probabilities)
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={score:.3f})",
            ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Random ranking",
        ))
        roc_fig.update_layout(
            title="ROC curves — held-out test set",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate (recall)",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

        st.markdown("#### Precision–recall curves")
        pr_fig = go.Figure()
        for name, probabilities in probabilities_by_model.items():
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
            ap = average_precision_score(y_test, probabilities)
            pr_fig.add_trace(go.Scatter(
                x=recall_curve, y=precision_curve, mode="lines",
                name=f"{name} (AP={ap:.3f})",
            ))
        prevalence = float(np.mean(y_test))
        pr_fig.add_hline(
            y=prevalence,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Positive prevalence = {prevalence:.3f}",
        )
        pr_fig.update_layout(
            title="Precision–recall curves — held-out test set",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        )
        st.plotly_chart(pr_fig, use_container_width=True)

        st.markdown("#### Export results")
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download evaluation table (CSV)",
            data=csv_bytes,
            file_name="predictive_maintenance_evaluation.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download selected model pipeline",
            data=io.BytesIO(__import__("pickle").dumps(trained["trained_models"][selected_eval_model])).getvalue(),
            file_name=f"{selected_eval_model.lower().replace(' ', '_')}_pipeline.pkl",
            mime="application/octet-stream",
        )

        st.warning(
            "The threshold slider is for visual analysis. Do not choose a final operating threshold "
            "by optimizing metrics on the test set; choose it using validation data, then evaluate once on test data."
        )

with tab_api_predict:
    st.subheader("Machine prediction through FastAPI")
    st.write(
        "Enter a machine's operating conditions to request a failure prediction from the trained model service. "
        "Train the model in Preparation & Training and keep the FastAPI service running."
    )

    api_url = st.text_input(
        "FastAPI base URL",
        value=API_BASE_URL,
        key="prediction_api_base_url",
        help="Local default: http://127.0.0.1:8000",
    ).rstrip("/")

    # Model publication happens automatically after successful training. This
    # page is kept focused on prediction and a simple refresh control.

    refresh_col, status_col = st.columns([1, 2])
    with refresh_col:
        refresh_models = st.button(
            "Refresh model list from API",
            key="refresh_api_model_list",
            use_container_width=True,
        )
    with status_col:
        st.caption("API documentation is available at the base URL + /docs.")

    if refresh_models:
        try:
            response = requests.get(f"{api_url}/models", timeout=3)
            response.raise_for_status()
            api_payload = response.json()
            st.session_state["api_model_catalog"] = api_payload
            if api_payload.get("models"):
                st.success(f"Connected. {len(api_payload['models'])} model(s) available.")
            else:
                st.warning(api_payload.get("message", "The API has no published models yet."))
        except requests.RequestException as exc:
            st.session_state.pop("api_model_catalog", None)
            st.error(
                f"Cannot reach FastAPI at {api_url}. Start the API in another terminal. Details: {exc}"
            )

    api_payload = st.session_state.get("api_model_catalog", {})
    api_models = api_payload.get("models", []) if isinstance(api_payload, dict) else []

    if not api_models:
        st.info(
            "No trained models are available from the prediction service yet. Complete a successful AI4I training run, "
            "then click 'Refresh model list from API'."
        )
    else:
        model_names = [item["name"] for item in api_models]
        default_model = api_payload.get("default_model")
        default_index = model_names.index(default_model) if default_model in model_names else 0

        with st.form("api_machine_prediction_form"):
            selected_api_model = st.selectbox(
                "Model",
                model_names,
                index=default_index,
            )
            st.markdown("Enter the machine's current operating conditions.")
            c1, c2, c3 = st.columns(3)
            with c1:
                machine_type = st.selectbox("Type", ["L", "M", "H"], index=1)
                air_temp = st.number_input(
                    "Air temperature [K]", min_value=1.0, max_value=1000.0,
                    value=298.1, step=0.1, format="%.1f"
                )
            with c2:
                process_temp = st.number_input(
                    "Process temperature [K]", min_value=1.0, max_value=1000.0,
                    value=308.6, step=0.1, format="%.1f"
                )
                rotational_speed = st.number_input(
                    "Rotational speed [rpm]", min_value=1, max_value=100000,
                    value=1551, step=1
                )
            with c3:
                torque = st.number_input(
                    "Torque [Nm]", min_value=0.0, max_value=10000.0,
                    value=42.8, step=0.1, format="%.1f"
                )
                tool_wear = st.number_input(
                    "Tool wear [min]", min_value=0, max_value=100000,
                    value=120, step=1
                )

            prediction_threshold = st.slider(
                "Decision threshold",
                min_value=0.01, max_value=0.99,
                value=float(api_payload.get("default_threshold", 0.5)),
                step=0.01,
                help="The model returns a failure probability. Class 1 is returned if it meets/exceeds this threshold.",
            )
            predict_submitted = st.form_submit_button(
                "Predict machine failure",
                type="primary",
                use_container_width=True,
            )

        if predict_submitted:
            request_body = {
                "model_name": selected_api_model,
                "threshold": float(prediction_threshold),
                "machine": {
                    "Type": machine_type,
                    "Air temperature [K]": float(air_temp),
                    "Process temperature [K]": float(process_temp),
                    "Rotational speed [rpm]": int(rotational_speed),
                    "Torque [Nm]": float(torque),
                    "Tool wear [min]": int(tool_wear),
                },
            }
            try:
                with st.spinner("Sending machine data to FastAPI..."):
                    response = requests.post(
                        f"{api_url}/predict",
                        json=request_body,
                        timeout=15,
                    )
                if not response.ok:
                    detail = response.json().get("detail", response.text)
                    st.error(f"Prediction API returned HTTP {response.status_code}: {detail}")
                else:
                    result = response.json()
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Predicted class", str(result["prediction"]))
                    metric2.metric("Failure probability", f"{result['failure_probability']:.2%}")
                    metric3.metric("Threshold", f"{result['threshold']:.2f}")
                    if result["prediction"] == 1:
                        st.error("Prediction: **Machine failure (class 1)**")
                    else:
                        st.success("Prediction: **No machine failure (class 0)**")
                    st.caption(
                        f"Model: {result['model_name']} · The prediction is a model estimate, "
                        "not a guarantee about the physical machine."
                    )
                    with st.expander("API response details"):
                        st.json(result)
            except requests.RequestException as exc:
                st.error(f"Could not call the prediction API: {exc}")


with tab_about:
    st.subheader("Project scope")
    st.markdown(
        """
        **Business problem:** identify operating conditions associated with machine failure using binary classification.

        **Current functionality:**
        - Automatic download of the AI4I 2020 dataset from UCI or CSV upload.
        - Data preview, quality summary and exploratory visualizations.
        - Fixed training target (Machine failure, positive class 1); feature selection for non-AI4I-compatible CSVs; stratified splitting, imputation, categorical encoding and optional scaling.
        - Five classifiers: Logistic Regression, Decision Tree, KNN, Random Forest and XGBoost.
        - Cross-validation, optional randomized hyperparameter search, and optional random oversampling inside training folds.
        - Test-set metric comparison, confusion matrix, ROC curves, precision–recall curves and a decision-threshold slider.

        **Implemented in the API phase:** FastAPI health/model-list/prediction endpoints and a Streamlit prediction tab that calls FastAPI. **Deferred:** SHAP and LIME explanations (Module 5).

        **Important limitations:** AI4I 2020 is synthetic data. Good test-set results on this dataset do not guarantee equivalent performance on real industrial equipment. The relative costs of missed failures and false alarms should guide model and threshold selection.
        """
    )
