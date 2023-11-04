import os
import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/scripts"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("build_dataset")


def get_tokens(data: pd.Series) -> Tuple[np.ndarray, List[int]]:
    """Токинезация текста, без батчей

    Args:
        data (pd.Series) данные с описанием обязанностей

    Returns:
        (Tuple[np.ndarray, List[int]]):
            * токенезированные данные с длинной 300
            * валидные индексы (пропускаем пустые)
    """

    data = data.to_list()
    result = []
    idx = []
    curr_id = 0
    for vac in tqdm(data):
        if vac is not None:
            result.append(tokenizer.encode(vac, max_length=300, truncation=True, padding="max_length"))
            idx.append(curr_id)
        curr_id += 1

    return np.array(result), idx


def get_time_cohortes(time_data: pd.Series, idx: List[int]) -> np.ndarray:
    """Получение когорт по востребованности

    Args:
        salary (pd.Series) данные о времени
        idx (List[int]) валидные индексы

    Returns:
        (np.ndarray): когорты зарплаты
    """
    results = []
    for curr_id in tqdm(idx):
        if time_data[curr_id].days < 10:
            results.append(0)
        elif 10 < time_data[curr_id].days < 20:
            results.append(1)
        else:
            results.append(2)
    return np.array(results)


if __name__ == "__main__":
    hh_data = pickle.load(open("/scripts/data/hh_data", "rb"))
    hh_data["onboard"] = pd.to_datetime(dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+0300")) - pd.to_datetime(
        hh_data["datetime"]
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    responsibility, idx = get_tokens(hh_data["responsibility"])
    print(f"Валидных данных: {len(idx)}")

    time_labels = get_time_cohortes(hh_data["onboard"], idx)
    print(f"Валидных данных: {len(idx)}")

    X_train, X_test, y_train, y_test = train_test_split(responsibility, time_labels, test_size=0.2, random_state=342)
    print(f"Обучающих объектов: {X_train.shape[0]}, тестовых: {X_test.shape[0]}")

    pickle.dump((X_train, X_test, y_train, y_test), open("/scripts/data/dataset", "wb"))
    with mlflow.start_run():
        mlflow.log_param("train_vectors", len(X_train))
        mlflow.log_param("test_vectors", len(X_test))
        mlflow.log_param("vector_length", X_test.train.shape[0])
