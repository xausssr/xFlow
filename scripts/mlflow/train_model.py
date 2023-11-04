import os
import pickle
import time
from xgboost import XGBClassifier

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/scripts"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

if __name__ == "__main__":
    X_train, _, y_train, _ = pickle.load(open("/scripts/data/dataset", "rb"))
    start = time.time()
    model = XGBClassifier(
        n_estimators=50,
        max_depth=10,
        learning_rate=1e-3,
        objective="multi:softmax",
        num_class=5,
    )
    model.fit(X_train, y_train)
    end = time.time() - start
    pickle.dump(model, open("/scripts/model/model", "wb"))
    with mlflow.start_run():
        mlflow.log_param("train_time", end)
