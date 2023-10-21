import pickle

from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = pickle.load(open("/scripts/airflow/data/dataset", "rb"))
    model = pickle.load(open("/scripts/airflow/model/model", "rb"))

    print("Результаты (ROC AUC):")
    print(f"\tобучение {roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovo'):.3f}")
    print(f"\tтест     {roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo'):.3f}")
