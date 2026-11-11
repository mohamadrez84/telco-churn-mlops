# Telco Customer Churn MLOps

An end-to-end churn classification project using the IBM Telco Customer Churn sample. It compares four classifiers on two feature sets, tracks experiments in MLflow, and serves the selected model through a Flask API.

[Model report](docs/model-report.md) · [راهنمای فارسی](docs/README.fa.md)

## Results

| Selected model | Validation F1 | Test F1 | Test precision | Test recall | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost with engineered features | 0.6435 | 0.6283 | 0.5352 | 0.7607 | 0.8531 |

The pipeline selects the model and feature set by validation F1. Only the selected model is scored on the internal test split. See the [candidate comparison](results/model_comparison.csv), [experiment snapshot](results/experiment_summary.json), and [confusion matrix](results/confusion_matrix.png). No independent cohort was available for external validation.

## Run the API

From the repository root, using Python 3.13:

```sh
python -m pip install -r requirements-serving.txt
python -m deployment.app
```

Or build and run the container:

```sh
docker build -t telco-churn-api .
docker run --rm -p 127.0.0.1:8000:8000 telco-churn-api
```

```sh
curl http://127.0.0.1:8000/health
curl -H "Content-Type: application/json" --data-binary "@examples/customer.json" http://127.0.0.1:8000/predict
```

Use `curl.exe` in Windows PowerShell. The example includes every raw field. The API returns a class, label, and probability. Missing or invalid inputs return HTTP 400.

## Reproduce training

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m run_pipeline
mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

Each execution logs eight candidates and exports the latest metrics, parameters, dataset hash, and test confusion matrix under `results/`. The selected model bundle under `models/` includes the fitted scaler, feature order, and training-derived threshold. The [model report](docs/model-report.md#mlflow-on-another-machine) explains how to keep the complete run history and artifacts together.

## Score an independent cohort

```sh
python -m evaluate_external path/to/independent-cohort.csv
```

Use the raw fields in the [API example](examples/customer.json) plus unique `CustomerID` values and observed binary `Churn Value` labels. CSV and XLSX are supported. The command rejects IDs overlapping the IBM sample and prints aggregate metrics for the existing model. Use a separately collected cohort, since an ID check alone cannot establish data independence.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/v1`, `data/v2`, `data/v3` | Raw sample, encoded data, and engineered features |
| `src/` and `run_pipeline.py` | Preprocessing, training, evaluation, and tracking |
| `models/` and `results/` | Selected model and evaluation artifacts |
| `deployment/` and `Dockerfile` | Flask API served by Waitress |
| `tests/` and `.github/workflows/ci.yml` | Regression tests and container HTTP smoke test |

Data source: [IBM Cognos Analytics Telco Customer Churn sample](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113?hlmlt=VT).
