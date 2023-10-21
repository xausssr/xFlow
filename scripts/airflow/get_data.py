import requests
import json
import time
import pandas as pd
import datetime as dt
import pickle
from typing import List, Any, Dict


class ExtendedDict(dict):
    def batch_add(self, batch: Dict[str, Any]) -> None:
        """Обновление словаря списков по ключам - каждому списку добавляется значение с помощью append.
        Аналогично pd.concat, но для обычных dict.

        Обновляемые ключи должны содержать плоские списки!

        Args:
            target_dict (Dict[str, List[Any]]):
            batch (Dict[str, Any]): данные для добавление в целевой словарь
        """
        for k, v in batch.items():
            self[k].append(v)


class VacanciesDumper:
    """Класс для дампа вакансий с hh.ru"""

    def __init__(
        self, url: str, search_queries: List[str], area: str, prev_data: pd.DataFrame = None, timeout: int = 2
    ) -> None:
        """Инициализация класса

        Args:
            url (str): эндпоинт api, используется для унификации функций.
            search_queries (List[str]): список запросов (текстовых).
            area (str): id региона (см. табличку `areas`).
            prev_data (pd.DataFrame, Optional): если есть предыдущие данные, то обогощаем их, тут же MLOps,
                повторяемость, накопление, все дела. Defaults to None.
            timeout (int): после каждого запроса пауза в <timeout> мс, что б не банили. Defaults to 2.

        Returns:
            pd.DataFrame: содержит поля:
                id - если до прода дойдет - отдавать карточки работы😐 [id]
                name - название вакансии [vac_name]
                salary - зарплата:
                    from - от [salary_from]
                    to - до [salary_to]
                published_at - размещена [datetime]
                type [id] - если открта (оценка по времени поиска специалиста) [state]
                employer - работодатель:
                    id - id работодателя (потом можно дернуть интересующих с полной информацией) [employer_id]
                    name - имя работодателя [employer_name]
                snippet - основные данные
                    requirement - что нужно соискателю [requirement]
                    responsibility - что будет делать [responsibility]
                schedule [id] - тип работы (офис, удаленка и т.д.) [schedule]
                experience [name] - требуемый опыт [experience]
                employment [id] - тип занятости [employment]
                professional_roles - роли на работе (опять же, может пригодится) - хранить будем прям строкой [roles]
        """

        self.page = 0
        self.pages = None
        if prev_data is None:
            prev_data = pd.DataFrame(self.get_template())
        self.data = prev_data
        self.url = url
        self.area = area
        self.search_queries = search_queries
        self.timeout = timeout

        self.new_vacancies = 0
        self.drop_vacancies = 0

    @staticmethod
    def get_template() -> Dict[str, List[str]]:
        return ExtendedDict(
            {
                "id": [],
                "vac_name": [],
                "salary_from": [],
                "salary_to": [],
                "datetime": [],
                "state": [],
                "employer_id": [],
                "employer_name": [],
                "requirement": [],
                "responsibility": [],
                "schedule": [],
                "experience": [],
                "employment": [],
                "roles": [],
            }
        )

    @staticmethod
    def __log(message) -> None:
        ts = dt.datetime.now().strftime("%d.%m.%y %H:%M:%S")
        print(f"[{ts}] {message}")

    def __dump_page(self, query) -> None:
        """Дамп одной страницы"""

        with requests.get(self.url, {"area": self.area, "per_page": 100, "text": query, "page": self.page}) as req:
            raw_data = json.loads(req.content.decode())

        _temp_dict = self.get_template()
        for vac in raw_data["items"]:
            if vac["id"] in self.data["id"]:
                self.drop_vacancies += 1
            else:
                salary = {} if vac["salary"] is None else vac["salary"]
                employer = {} if vac["employer"] is None else vac["employer"]
                snippet = {} if vac["snippet"] is None else vac["snippet"]
                schedule = {} if vac["schedule"] is None else vac["schedule"]
                state = {} if vac["type"] is None else vac["type"]
                experience = {} if vac["experience"] is None else vac["experience"]
                employment = {} if vac["employment"] is None else vac["employment"]

                _temp_dict.batch_add(
                    {
                        "id": vac["id"],
                        "vac_name": vac["name"],
                        "salary_from": salary.get("from", None),
                        "salary_to": salary.get("from", None),
                        "datetime": vac["published_at"],
                        "state": state.get("id", None),
                        "employer_id": employer.get("id", None),
                        "employer_name": employer.get("id", None),
                        "requirement": snippet.get("requirement", None),
                        "responsibility": snippet.get("responsibility", None),
                        "schedule": schedule.get("id", None),
                        "experience": experience.get("name", None),
                        "employment": employment.get("id", None),
                        "roles": vac["professional_roles"],
                    }
                )
                self.new_vacancies += 1

        self.data = pd.concat([self.data, pd.DataFrame(_temp_dict)], axis=0, ignore_index=True)

        self.page += 1
        if self.pages is None:
            self.pages = int(raw_data["pages"])

    def __dump_query(self, query) -> None:
        """Дамп одного запроса из `search_queries`"""

        self.page = 0
        self.__dump_page(query)
        time.sleep(self.timeout)

        while self.page < self.pages:
            self.__dump_page(query)
            self.__log(f'Запрос "{query}", страница {self.page}/{self.pages}')
            time.sleep(self.timeout)

    def dump(self) -> pd.DataFrame:
        """Начало дампа вакансий"""

        self.__log(f"Начинаю дамп {len(self.search_queries)} запросов")
        for query in self.search_queries:
            self.__dump_query(query)
            self.pages = None
            self.__log(f'Запрос "{query}" обработан!')

        self.__log(f"Дамп завершен! Новых записей: {self.new_vacancies}, отброшено: {self.drop_vacancies}")
        return self.data


if __name__ == "__main__":
    dumper = VacanciesDumper(
        "https://api.hh.ru/vacancies",
        search_queries=["it", "ml engineer", "data engineer", "computer vision", "nlp", "neural networks"],
        area="1",
    )

    hh_data = dumper.dump()
    pickle.dump(hh_data, open("/scripts/airflow/data/hh_data", "wb"))
