# AI4I 2020 Predictive Maintenance

An interactive binary-classification project for predicting machine failure from operating conditions. It combines a Streamlit dashboard for data exploration, preprocessing, model training and evaluation with a FastAPI service for machine-level predictions.

> **Scope note:** Module 5 now includes global and local SHAP explanations, LIME local explanations, a Streamlit XAI tab, and a FastAPI `POST /explain` endpoint. AI4I 2020 is a synthetic dataset, so results do not guarantee equivalent performance on real industrial equipment.

## Contents

- [Problem and dataset](#problem-and-dataset)
- [Features and data-leakage controls](#features-and-data-leakage-controls)
- [Implemented functionality](#implemented-functionality)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Run the application on Windows](#run-the-application-on-windows)
- [Train models and publish them to the API](#train-models-and-publish-them-to-the-api)
- [Explain predictions with SHAP and LIME](#explain-predictions-with-shap-and-lime)
- [FastAPI endpoints](#fastapi-endpoints)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Problem and dataset

The task is to classify whether a machine failure occurs (`Machine failure = 1`) or not (`Machine failure = 0`) based on machine operating conditions.

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository:

- Dataset page: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- The Streamlit application can download the dataset automatically through `ucimlrepo` or accept a CSV upload.
- The original AI4I dataset contains 10,000 observations. The project notebook found 339 failure observations (3.39%), so the positive class is a minority class.

## Features and data-leakage controls

For the AI4I task, the target is fixed to `Machine failure` and the positive class is fixed to `1`.

The six approved model predictors are:

1. `Type`
2. `Air temperature [K]`
3. `Process temperature [K]`
4. `Rotational speed [rpm]`
5. `Torque [Nm]`
6. `Tool wear [min]`

The identifier columns `UDI` and `Product ID` are not used as predictors. The failure-mode indicators `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are excluded from the AI4I model features because they reveal information closely tied to the target and would cause target leakage. Column names returned by the UCI Python client are normalized to the canonical names above.

The training tab fixes the target and positive class for this project. When an AI4I-compatible dataset is detected, it also fixes the six approved features.

## Implemented functionality

### Streamlit dashboard

- Automatic UCI dataset download or CSV upload.
- Dataset preview, dimensions, data types, missing-value counts, duplicate-row count and descriptive statistics.
- Interactive exploratory visualizations, including class distributions, numeric distributions, correlation heatmaps and positive-class rates by category.
- Configurable train/test split, cross-validation folds and random seed.
- Dtype-aware preprocessing: median imputation and optional scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Five classifiers: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Random Forest and XGBoost.
- Stratified cross-validation and optional `RandomizedSearchCV` hyperparameter tuning.
- Optional random oversampling inside the training pipeline, rather than before train/test splitting.
- Test-set comparison using accuracy, precision, recall, F1, ROC-AUC, Average Precision, trapezoidal PR-AUC and confusion-matrix counts.
- Confusion matrix, ROC curves, precision-recall curves, classification report, threshold exploration and CSV/model downloads.
- Machine Prediction tab that calls the FastAPI service.
- Explainability tab with global SHAP feature importance/beeswarm plots and local SHAP waterfall plus LIME surrogate explanations for a selected held-out observation.

### FastAPI service

- Publishes trained AI4I-compatible pipelines into a local model registry after successful training in Streamlit.
- Lists available models and identifies a default model.
- Validates machine feature inputs and returns the predicted class and estimated failure probability at the selected threshold.
- Provides interactive API documentation through Swagger UI.
- `POST /explain` returns local SHAP contributions and LIME feature-condition weights for a machine input, using the local AI4I CSV as the explanation background.

## Project structure

```text
predictive-maintenance/
├── app.py                         # Streamlit UI, data, training, evaluation, XAI and API client
├── xai.py                         # Shared SHAP/LIME helpers for the app, API and notebook
├── requirements.txt               # Python dependencies
├── README.md
├── .gitignore
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI service
├── dataset/  
│   └──  ai4i2020.csv 
├── models/
│   └── api_serving/
│       ├── .gitkeep               # Keeps empty registry directory in Git
│       ├── catalog.json           # Generated after successful AI4I training
│       └── *.joblib               # Generated fitted pipelines
├── notebooks/
│   └── ai4i_predictive_maintenance.ipynb  # Phases 1–6, including SHAP/LIME
└── tests/
    └── test_xai.py                # XAI schema and prediction consistency tests
```

`catalog.json` and the `.joblib` files under `models/api_serving/` are generated artifacts. With the recommended `.gitignore`, they are not committed by default; after cloning the repository, run an AI4I training session in Streamlit to regenerate the local registry. If you choose to commit trained artifacts instead, first pin the compatible dependency versions and ensure the files meet GitHub's file-size limits.

## Requirements

- Windows 10/11, macOS, or Linux.
- Python compatible with the package versions in `requirements.txt`. Python 3.12 is a reasonable default for this project.
- Internet access for dependency installation and automatic UCI download.
- Git for version control and publishing to GitHub.

The current `requirements.txt` includes Streamlit, Pandas, NumPy, Plotly, scikit-learn, imbalanced-learn, XGBoost, `ucimlrepo`, joblib, Requests, FastAPI, Uvicorn, Matplotlib, SHAP and LIME.

## Run the application on Windows

### 1. Clone the repository

Open PowerShell in the directory where you want to store the project, then run:

```powershell
git clone https://github.com/mohamedborhen/ai4i-predictive-maintenance.git
cd ai4i-predictive-maintenance
```

### 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Start Streamlit

In the first PowerShell terminal, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

### 5. Load data and train models

1. In the sidebar, choose **AI4I 2020 (UCI, automatic download)**. If the UCI download fails, choose **Upload a CSV file** and upload `ai4i2020.csv`.
2. Open **Preparation & Training**. The target is fixed to `Machine failure`, and positive class `1` is used.
3. For AI4I, verify that the six approved predictors are used. The failure-mode labels and identifiers must not be model predictors.
4. Select one or more classifiers. Start with hyperparameter tuning off to verify the workflow, because tuning can require many model fits.
5. Choose the split ratio, cross-validation folds and imbalance strategy, then click **Train selected models**.
6. Successful AI4I training automatically writes the fitted pipelines and `catalog.json` under `models/api_serving/`.
7. Open **Evaluation** to compare models and inspect metrics, curves, the confusion matrix and threshold behaviour.

### 6. Start FastAPI in a second terminal

Keep Streamlit running. Open a second PowerShell terminal in the `ai4i-predictive-maintenance` project directory and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

The local API binds to `127.0.0.1`, which is suitable for local development.

### 7. Use machine prediction

Return to Streamlit, open **Machine Prediction (FastAPI)**, and click **Refresh model list from API**. Select an available model, enter the six machine measurements, choose a decision threshold and click **Predict machine failure**.

The API returns class `0` (no machine failure) or class `1` (machine failure) and a failure probability. The probability is a model estimate, not a guarantee of physical machine condition.

## Train models and publish them to the API

The API registry is created only after successful training from Streamlit with all of these conditions:

- Target: `Machine failure`.
- Positive class: `1`.
- Predictors: exactly the six approved AI4I features listed above.
- At least one fitted model.


Run the commands from the project root. The Streamlit app and Uvicorn process must use the same project directory. A successful HTTP response from `/models` does not guarantee that models exist; an empty registry is reported as `ready: false`.

## FastAPI endpoints

Once Uvicorn is running:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service overview |
| `GET` | `/health` | API status and registry readiness |
| `GET` | `/models` | List published models and default settings |
| `POST` | `/predict` | Return prediction class and failure probability |
| `POST` | `/explain` | Return local SHAP contributions and LIME feature-condition weights |
| `GET` | `/docs` | Interactive Swagger API documentation |

Swagger UI: <http://127.0.0.1:8000/docs>

Example `POST /predict` body (the `model_name` must match a value returned by `GET /models`):

```json
{
  "model_name": "KNN",
  "threshold": 0.5,
  "machine": {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 120
  }
}
```


### Explain a prediction through FastAPI

Use `POST /explain` with the registered model name and the six original feature values. The endpoint returns the estimated failure probability, one SHAP contribution per original feature, and LIME's local feature-condition weights. It uses a reproducible sample of rows from `dataset/ai4i2020.csv` as the background distribution, so that file must be present on the API host.

Example request:

```json
{
  "model_name": "Decision Tree",
  "threshold": 0.5,
  "lime_num_samples": 1500,
  "machine": {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 120
  }
}
```

The endpoint can be slow because model-agnostic SHAP and LIME require repeated model predictions. The response is a local explanation, not a causal account. Install the updated `requirements.txt` and restart FastAPI after updating the code.

## Explain predictions with SHAP and LIME

In Streamlit, train an AI4I model and open **Explainability (SHAP + LIME)**. Select a model, a training-background size, the number of held-out rows for global SHAP, and a held-out row for local explanation. The tab shows mean absolute SHAP importance, a SHAP beeswarm plot, a local SHAP waterfall plot, and LIME local feature-condition weights.

The notebook's final Phase 6 cells provide the same analysis using the model selected by cross-validated Average Precision. The notebook cells are appended after the existing evaluation cells; previous cells are retained.

`POST /explain` returns a JSON response with the estimated failure probability, one SHAP contribution per original feature, and LIME feature-condition weights. The endpoint samples the local `dataset/ai4i2020.csv` as its explanation background, so the dataset must be present on the API host. Model-agnostic SHAP and LIME require repeated model predictions and may take longer than `/predict`.

Example request:

```json
{
  "model_name": "Decision Tree",
  "threshold": 0.5,
  "lime_num_samples": 1500,
  "machine": {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 120
  }
}
```

Positive SHAP contributions move the explained output toward a higher failure probability relative to the SHAP baseline; negative contributions move it lower. LIME weights describe a local surrogate and can differ from SHAP. Neither method establishes causation.

## Limitations and responsible use

- The AI4I dataset is synthetic. Evaluation results on it should not be treated as proof of performance on real machines.
- The dataset is highly imbalanced; review failure-class recall, precision, F1 and PR-AUC alongside accuracy.
- The decision threshold affects missed failures and false alarms. Choose a deployment threshold using validation data and operational costs, not by repeatedly optimizing the test set.
- Oversampling, when selected, is applied within the model-training pipeline rather than to the test set.
- SHAP and LIME explain model behaviour, not causal relationships; explanations can vary with the background data and local surrogate sampling.
- The FastAPI service is configured for local development. A public deployment needs a separately hosted API, HTTPS, authentication, and suitable operational controls. The default `127.0.0.1` API URL refers to the same computer and will not reach a remote API from a cloud-hosted Streamlit app.
