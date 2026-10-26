from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from xai import FEATURES as XAI_FEATURES, explanation_payload, validate_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "api_serving"
CATALOG_PATH = MODEL_DIR / "catalog.json"

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0.0",
    description=(
        "Prediction API for the AI4I 2020 binary machine-failure classifier, "
        "including local SHAP and LIME explanations."
    ),
)


class MachineFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    machine_type: Literal["L", "M", "H"] = Field(alias="Type")
    air_temperature: float = Field(alias="Air temperature [K]", gt=0, le=1000)
    process_temperature: float = Field(alias="Process temperature [K]", gt=0, le=1000)
    rotational_speed: int = Field(alias="Rotational speed [rpm]", gt=0, le=100000)
    torque: float = Field(alias="Torque [Nm]", ge=0, le=10000)
    tool_wear: int = Field(alias="Tool wear [min]", ge=0, le=100000)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1, description="Name of a previously trained AI4I model")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    machine: MachineFeatures


class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1, description="Name of a previously trained AI4I model")
    machine: MachineFeatures
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    lime_num_samples: int = Field(default=1500, ge=500, le=5000)


def read_catalog() -> dict[str, Any] | None:
    if not CATALOG_PATH.is_file():
        return None
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read model catalog: {exc}") from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        raise HTTPException(status_code=500, detail="Model catalog is malformed.")
    return catalog


@lru_cache(maxsize=16)
def load_pipeline(artifact_path: str, modified_ns: int):
    """Load one locally-created trusted pipeline; mtime invalidates cache on update."""
    return joblib.load(artifact_path)


def resolve_artifact(entry: dict[str, Any]) -> Path:
    artifact_name = entry.get("artifact", "")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
        raise HTTPException(status_code=500, detail="Invalid artifact name in model catalog.")
    if not artifact_name.endswith(".joblib"):
        raise HTTPException(status_code=500, detail="Unsupported model artifact extension.")
    artifact_path = (MODEL_DIR / artifact_name).resolve()
    if artifact_path.parent != MODEL_DIR.resolve() or not artifact_path.is_file():
        raise HTTPException(status_code=500, detail=f"Model artifact not found: {artifact_name}")
    return artifact_path


@app.get("/", tags=["System"])
def root():
    return {
        "service": "AI4I Predictive Maintenance API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "prediction": "POST /predict",
        "explanation": "POST /explain",
    }


@app.get("/debug/registry", tags=["System"], include_in_schema=False)
def debug_registry():
    """Local troubleshooting endpoint: show which registry path this API reads."""
    files = sorted(path.name for path in MODEL_DIR.iterdir()) if MODEL_DIR.is_dir() else []
    catalog = None
    catalog_error = None
    if CATALOG_PATH.is_file():
        try:
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            catalog_error = f"{type(exc).__name__}: {exc}"
    return {
        "project_root": str(PROJECT_ROOT),
        "registry_directory": str(MODEL_DIR),
        "catalog_path": str(CATALOG_PATH),
        "catalog_exists": CATALOG_PATH.is_file(),
        "registry_directory_exists": MODEL_DIR.is_dir(),
        "registry_files": files,
        "catalog_model_count": len(catalog.get("models", [])) if isinstance(catalog, dict) else 0,
        "catalog_error": catalog_error,
    }


@app.get("/health", tags=["System"])
def health():
    catalog = read_catalog()
    model_count = len(catalog["models"]) if catalog else 0
    return {
        "status": "ok",
        "model_registry_ready": model_count > 0,
        "available_models": model_count,
    }


@app.get("/models", tags=["Models"])
def list_models():
    catalog = read_catalog()
    if catalog is None:
        return {
            "ready": False,
            "message": (
                "No AI4I models have been published. Train models in the Streamlit app "
                "with the AI4I target and six approved predictors first."
            ),
            "default_model": None,
            "default_threshold": 0.5,
            "models": [],
        }
    return {
        "ready": bool(catalog.get("models")),
        "message": None if catalog.get("models") else "No models are registered.",
        "default_model": catalog.get("default_model"),
        "default_threshold": float(catalog.get("default_threshold", 0.5)),
        "features": catalog.get("features", []),
        "models": [
            {
                "name": item.get("name"),
                "cross_validated_average_precision": item.get("cross_validated_average_precision"),
            }
            for item in catalog.get("models", [])
        ],
    }


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest):
    catalog = read_catalog()
    if catalog is None or not catalog.get("models"):
        raise HTTPException(
            status_code=409,
            detail="No trained AI4I model is available. Train and publish models from Streamlit first.",
        )

    entries = {entry.get("name"): entry for entry in catalog["models"]}
    if request.model_name not in entries:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' is not registered. Call GET /models to list available models.",
        )

    features = catalog.get("features", [])
    required_features = {
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    }
    if set(features) != required_features or len(features) != 6:
        raise HTTPException(
            status_code=500,
            detail="Model catalog feature schema does not match the six expected AI4I predictors.",
        )

    try:
        values = request.machine.model_dump(by_alias=True)
        row = pd.DataFrame([{column: values[column] for column in features}], columns=features)
        artifact_path = resolve_artifact(entries[request.model_name])
        pipeline = load_pipeline(str(artifact_path), artifact_path.stat().st_mtime_ns)
        probabilities = pipeline.predict_proba(row)[0]
        classes = list(pipeline.classes_)
        if 1 not in classes:
            raise ValueError("The saved classifier does not contain positive class 1.")
        failure_probability = float(probabilities[classes.index(1)])
        prediction = int(failure_probability >= request.threshold)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    ap = entries[request.model_name].get("cross_validated_average_precision")
    return {
        "model_name": request.model_name,
        "prediction": prediction,
        "prediction_label": "Machine failure" if prediction == 1 else "No machine failure",
        "failure_probability": failure_probability,
        "threshold": request.threshold,
        "positive_class": 1,
        "cross_validated_average_precision": ap,
        "note": "Prediction is a model estimate, not a guarantee of physical machine condition.",
    }


@app.post("/explain", tags=["Explainability"])
def explain(request: ExplanationRequest):
    """Return local SHAP and LIME explanations for one machine prediction."""
    catalog = read_catalog()
    if catalog is None or not catalog.get("models"):
        raise HTTPException(
            status_code=409,
            detail="No trained AI4I model is available. Train and publish models from Streamlit first.",
        )

    entries = {entry.get("name"): entry for entry in catalog["models"]}
    if request.model_name not in entries:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' is not registered. Call GET /models to list available models.",
        )

    features = catalog.get("features", [])
    if set(features) != set(XAI_FEATURES) or len(features) != len(XAI_FEATURES):
        raise HTTPException(status_code=500, detail="Model catalog feature schema does not match the six expected AI4I predictors.")

    dataset_path = PROJECT_ROOT / "dataset" / "ai4i2020.csv"
    if not dataset_path.is_file():
        raise HTTPException(status_code=503, detail="The local AI4I dataset is required as an explanation background but was not found.")

    try:
        values = request.machine.model_dump(by_alias=True)
        row = pd.DataFrame([{column: values[column] for column in features}], columns=features)
        background_data = pd.read_csv(dataset_path)
        missing = [feature for feature in features if feature not in background_data.columns]
        if missing:
            raise ValueError(f"Background dataset is missing required features: {missing}")
        background = validate_features(
            background_data.loc[:, features].sample(n=min(60, len(background_data)), random_state=42)
        )
        artifact_path = resolve_artifact(entries[request.model_name])
        pipeline = load_pipeline(str(artifact_path), artifact_path.stat().st_mtime_ns)
        explanation = explanation_payload(
            pipeline, background, validate_features(row), lime_num_samples=request.lime_num_samples
        )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    explanation["model_name"] = request.model_name
    explanation["threshold"] = request.threshold
    explanation["prediction"] = int(explanation["failure_probability"] >= request.threshold)
    explanation["note"] = "This is a model explanation, not a causal explanation or a guarantee of machine condition."
    return explanation
