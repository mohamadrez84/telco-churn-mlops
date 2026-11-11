import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
import mlflow
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "Telco_Churn_Project"


def init_mlflow():
    tracking_dir = BASE_DIR / "tracking"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    use_local_store = not tracking_uri
    if use_local_store:
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    if use_local_store and mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(tracking_dir / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def save_confusion_matrix(matrix, destination, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        display = ConfusionMatrixDisplay(matrix, display_labels=("No Churn", "Churn"))
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title)
        fig.savefig(destination, bbox_inches="tight", dpi=140)
    finally:
        plt.close(fig)


def log_run(model_name, dataset_version, params, metrics, seed, bundle,
            confusion_matrix_path=None):
    run_name = f"{model_name}_{dataset_version}"
    with mlflow.start_run(run_name=run_name) as run, TemporaryDirectory() as temp_dir:
        mlflow.set_tags({
            "model_name": model_name,
            "dataset_version": dataset_version,
            "selected": str(confusion_matrix_path is not None).lower(),
        })
        mlflow.log_params({**params, "seed": seed, "dataset_version": dataset_version})
        mlflow.log_metrics(metrics)
        bundle_path = Path(temp_dir) / "model.pkl"
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(str(bundle_path), artifact_path="model")

        if confusion_matrix_path is not None:
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="confusion_matrix")

        print(f"[mlflow] {run_name} (run_id={run.info.run_id})")
