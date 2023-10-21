import pickle
from xgboost import XGBClassifier


if __name__ == "__main__":
    X_train, _, y_train, _ = pickle.load(open("/scripts/airflow/data/dataset", "rb"))
    model = XGBClassifier(
        n_estimators=50,
        max_depth=10,
        learning_rate=1e-3,
        objective="multi:softmax",
        num_class=5,
    )
    model.fit(X_train, y_train)
    pickle.dump(model, open("/scripts/airflow/model/model", "wb"))
