import os
import pickle

from sklearn.metrics import roc_auc_score

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/scripts"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("evaluate")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = pickle.load(open("/scripts/data/dataset", "rb"))
    model = pickle.load(open("/scripts/model/model", "rb"))

    train_roc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovo")
    test_roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo")
    print("Результаты (ROC AUC):")
    print(f"\tобучение {train_roc:.3f}")
    print(f"\tтест     {test_roc:.3f}")

    with mlflow.start_run():
        mlflow.log_param("train_roc", train_roc)
        mlflow.log_param("test_roc", test_roc)
