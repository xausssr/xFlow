import pickle
import datetime as dt
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import BertTokenizer


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


def get_salary_cohortes(salary: pd.Series, idx: List[int]) -> np.ndarray:
    """Получение когорт по зарплате

    Args:
        salary (pd.Series) данные о зарплате
        idx (List[int]) валидные индексы

    Returns:
        (np.ndarray): когорты зарплаты
    """
    results = []
    for curr_id in tqdm(idx):
        if np.isnan(salary[curr_id]):
            results.append(0)
        elif salary[curr_id] < 100_000:
            results.append(1)
        elif 100_000 < salary[curr_id] < 150_000:
            results.append(2)
        elif 150_000 < salary[curr_id] < 200_000:
            results.append(3)
        else:
            results.append(4)
    return np.array(results)


if __name__ == "__main__":
    hh_data = pickle.load(open("/scripts/airflow/data/hh_data", "rb"))
    hh_data["onboard"] = pd.to_datetime(dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S+0300")) - pd.to_datetime(
        hh_data["datetime"]
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    responsibility, idx = get_tokens(hh_data["responsibility"])
    salary = get_salary_cohortes(hh_data["salary_from"], idx)
    print(f"Валидных данных: {len(idx)}")

    X_train, X_test, y_train, y_test = train_test_split(responsibility, salary, test_size=0.2, random_state=342)
    print(f"Обучающих объектов: {X_train.shape[0]}, тестовых: {X_test.shape[0]}")

    pickle.dump((X_train, X_test, y_train, y_test), open("/scripts/airflow/data/dataset", "wb"))
