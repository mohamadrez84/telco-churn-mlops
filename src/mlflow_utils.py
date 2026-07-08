import os
import tempfile
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

TRACKING_DB = os.path.join(os.getcwd(), "mlflow.db")
EXPERIMENT_NAME = "Telco_Churn_Project"


def init_mlflow():
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(model_name, dataset_version, params, metrics, seed, confusion_matrix,
            model=None, class_names=("No Churn", "Churn")):
    run_name = f"{model_name}_{dataset_version}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_version", dataset_version)

        log_params = dict(params)
        log_params["seed"] = seed
        log_params["dataset_version"] = dataset_version
        mlflow.log_params(log_params)

        mlflow.log_metrics(metrics)

        # Confusion Matrix به‌عنوان یک artifact تصویری (قبلاً اصلاً ثبت نمی‌شد)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} | {dataset_version}")
        cm_path = os.path.join(tempfile.gettempdir(), f"cm_{model_name}_{dataset_version}.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # خود مدل هم ثبت می‌شود (قبلاً import شده بود ولی هرگز صدا زده نمی‌شد)
        if model is not None:
            try:
                model_type = type(model).__module__
                if "xgboost" in model_type:
                    from mlflow import xgboost as mlflow_xgboost
                    mlflow_xgboost.log_model(model, name="model")
                elif "catboost" in model_type:
                    from mlflow import catboost as mlflow_catboost
                    mlflow_catboost.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(model, name="model")
            except Exception as e:
                print(f"[mlflow_utils] ثبت مدل ممکن نشد: {e}")

        run_id = mlflow.active_run().info.run_id
        print(f"[mlflow_utils] Run ثبت شد -> {run_name} (run_id={run_id})")
        return run_id
