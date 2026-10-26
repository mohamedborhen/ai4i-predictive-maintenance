"""SHAP and LIME explanations for the AI4I predictive-maintenance pipelines.

The public functions accept the original six-feature DataFrame. Categorical Type
is encoded only inside the explainers; predictions always pass through the fitted
pipeline so its preprocessing and classifier remain authoritative.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_TO_CODE = {"L": 0, "M": 1, "H": 2}
CODE_TO_TYPE = {value: key for key, value in TYPE_TO_CODE.items()}


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the exact canonical feature order and valid values."""
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing required AI4I feature(s): {', '.join(missing)}")
    result = frame.loc[:, FEATURES].copy()
    result["Type"] = result["Type"].astype(str).str.upper()
    invalid_types = sorted(set(result["Type"].dropna()) - set(TYPE_TO_CODE))
    if invalid_types:
        raise ValueError(f"Unsupported machine Type value(s): {invalid_types}; expected L, M, or H.")
    for column in FEATURES[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.isna().any().any():
        raise ValueError("XAI inputs must not contain missing feature values.")
    return result


def encode_features(frame: pd.DataFrame) -> np.ndarray:
    """Represent the mixed-type inputs numerically for model-agnostic explainers."""
    validated = validate_features(frame)
    encoded = validated.copy()
    encoded["Type"] = encoded["Type"].map(TYPE_TO_CODE).astype(int)
    return encoded.to_numpy(dtype=float)


def decode_features(values: Any) -> pd.DataFrame:
    """Convert explainer-generated numeric rows back to the pipeline input schema."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, received {array.shape[1]}.")
    frame = pd.DataFrame(array, columns=FEATURES)
    type_codes = np.clip(np.rint(frame["Type"]).astype(int), 0, 2)
    frame["Type"] = type_codes.map(CODE_TO_TYPE)
    for column in FEATURES[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def predict_proba_encoded(pipeline: Any, values: Any) -> np.ndarray:
    """Predict both classes from numeric explainer rows through the fitted pipeline."""
    frame = decode_features(values)
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    if 0 not in classes or 1 not in classes:
        raise ValueError("The fitted model must expose both classes 0 and 1.")
    return probabilities[:, [classes.index(0), classes.index(1)]]


def compute_shap_explanations(
    pipeline: Any,
    background: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    max_evals: int | None = None,
):
    """Return a SHAP Explanation for positive-class (failure) probability.

    Uses a model-agnostic permutation explainer against the original six features,
    which keeps explanation labels interpretable across all supported pipelines.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError("SHAP is required for SHAP explanations. Install with `pip install shap`.") from exc

    background_values = encode_features(background)
    observation_values = encode_features(observations)
    if len(background_values) == 0 or len(observation_values) == 0:
        raise ValueError("SHAP requires non-empty background and observation data.")

    def failure_probability(values):
        return predict_proba_encoded(pipeline, values)[:, 1]

    masker = shap.maskers.Independent(background_values, max_samples=min(100, len(background_values)))
    explainer = shap.Explainer(
        failure_probability,
        masker=masker,
        algorithm="permutation",
        feature_names=FEATURES,
    )
    evaluation_budget = max_evals or (2 * len(FEATURES) + 1) * 5
    return explainer(observation_values, max_evals=evaluation_budget)


def compute_lime_explanation(
    pipeline: Any,
    background: pd.DataFrame,
    observation: pd.DataFrame,
    *,
    num_samples: int = 1500,
    num_features: int = 6,
):
    """Return a LIME Tabular explanation for one row and its class probabilities."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError("LIME is required for LIME explanations. Install with `pip install lime`.") from exc

    if len(observation) != 1:
        raise ValueError("LIME local explanation expects exactly one observation.")
    background_values = encode_features(background)
    observation_values = encode_features(observation)[0]
    explainer = LimeTabularExplainer(
        training_data=background_values,
        feature_names=FEATURES,
        class_names=["No machine failure (0)", "Machine failure (1)"],
        categorical_features=[0],
        categorical_names={0: ["L", "M", "H"]},
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        observation_values,
        lambda rows: predict_proba_encoded(pipeline, rows),
        labels=(1,),
        num_features=min(num_features, len(FEATURES)),
        num_samples=num_samples,
    )
    probabilities = predict_proba_encoded(pipeline, observation_values.reshape(1, -1))[0]
    return explanation, probabilities


def explanation_payload(
    pipeline: Any,
    background: pd.DataFrame,
    observation: pd.DataFrame,
    *,
    shap_max_evals: int | None = None,
    lime_num_samples: int = 1500,
) -> dict[str, Any]:
    """Build JSON-safe SHAP and LIME local explanations for one observation."""
    row = validate_features(observation).iloc[[0]]
    shap_result = compute_shap_explanations(
        pipeline, background, row, max_evals=shap_max_evals
    )
    lime_result, probabilities = compute_lime_explanation(
        pipeline, background, row, num_samples=lime_num_samples
    )
    shap_values = np.asarray(shap_result.values)
    if shap_values.ndim == 2:
        shap_values = shap_values[0]
    base_values = np.asarray(shap_result.base_values).reshape(-1)
    lime_items = [
        {"feature_rule": str(rule), "contribution": float(weight)}
        for rule, weight in lime_result.as_list(label=1)
    ]
    return {
        "prediction": int(probabilities[1] >= 0.5),
        "failure_probability": float(probabilities[1]),
        "positive_class": 1,
        "features": {
            feature: {
                "value": str(row.iloc[0][feature]) if feature == "Type" else float(row.iloc[0][feature]),
                "shap_value": float(shap_values[index]),
            }
            for index, feature in enumerate(FEATURES)
        },
        "shap_base_value": float(base_values[0]),
        "lime_explanation": lime_items,
        "explanation_note": (
            "SHAP values explain contributions to the estimated failure probability. "
            "LIME weights describe a local surrogate model. Neither method establishes causation."
        ),
    }
