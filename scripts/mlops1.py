from clearml.automation.controller import PipelineDecorator

@PipelineDecorator.component(cache=True, return_values=['path', 'labels', 'subs', 'sub_ids'], execution_queue="default")
def step_one():
   import os
   import pandas as pd
   from pathlib import Path
   from clearml import Dataset

   # Получите датасет по его ID
   dataset_id = "61a444098f324650a669793d884b2f17"
   dataset = Dataset.get(dataset_id)
   path = Path(dataset.get_local_copy())

   submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
   labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

   sub_ids = submission.index # the ids of the submission rows (useful later) [20000:40000]
   #gt_ids = labels.index # the ids of the labeled rows (useful later) [0:20000]

   #Добавим все файлы кроме 'sample_submission.csv', 'train_labels.csv'
   subs = [file for file in os.listdir(path) if file not in ['sample_submission.csv', 'train_labels.csv']]
   subs = sorted(subs)

   return path, labels, subs, sub_ids


@PipelineDecorator.component(cache=True, return_values=['ensemble_data'], execution_queue="default")
def step_two(path, subs, parameters):
   import pandas as pd

   # Конкатинирование значений предсказаний из файлов в единый датасет для передачи в train_split
   ensemble_data = pd.concat([pd.read_csv(path / subs[i], index_col='id') for i in range(parameters['number_files'])], axis=1)
   
   # Переименование дублирующихся столбцов
   ensemble_data.columns = [f"pred_{i}" for i in range(ensemble_data.shape[1])]

   return ensemble_data


@PipelineDecorator.component(return_values=['X_train, X_val, y_train, y_val'], execution_queue="default")
def step_three(ensemble_data, labels):
   from sklearn.model_selection import train_test_split

   X_train, X_val, y_train, y_val = train_test_split(ensemble_data[:20000], labels, test_size=0.2, random_state=42)

   return X_train, X_val, y_train, y_val


@PipelineDecorator.component(return_values=['y_pred_proba_val', 'y_pred_proba_test'], execution_queue="default")
def step_four(X_train, X_val, y_train, ensemble_data, parameters):
   from xgboost import XGBClassifier

   # Создание и обучение модели
   ensemble_model = XGBClassifier(
      max_depth = parameters['max_depth'],
      n_estimators = parameters['n_estimators'],
      learning_rate = parameters['learning_rate'],
      subsample = parameters['subsample'],
      colsample_bytree = parameters['colsample_bytree'],
      gamma = parameters['gamma'],
      reg_alpha = parameters['reg_alpha'],
      reg_lambda = parameters['reg_lambda'],
      objective = parameters['objective'],
      eval_metric = parameters['eval_metric'],
      random_state = parameters['random_state']
      )

   ensemble_model.fit(X_train, y_train)

   # Предсказание на валидационном наборе
   y_pred_proba_val = ensemble_model.predict_proba(X_val)[:, 1]

   # Предсказание на тестовом наборе
   # Создание DataFrame из массива blade[20000:]
   y_pred_proba_test = ensemble_model.predict_proba(ensemble_data[20000:])[:, 1]

   return y_pred_proba_val, y_pred_proba_test


@PipelineDecorator.component(return_values=['score'], execution_queue="default")
def step_five(y_val, y_pred_proba_val):
   from sklearn.metrics import log_loss

   # Расчет log_loss
   score = log_loss(y_val, y_pred_proba_val)
   return score


@PipelineDecorator.component(return_values=['blend'], execution_queue="default")
def step_six(y_pred_proba_test, sub_ids):
   import pandas as pd

   blend = pd.DataFrame({'pred': y_pred_proba_test}, index=sub_ids)
   return blend


parameters = {
   'max_depth' : 5,
   'n_estimators' : 100,
   'learning_rate' : 0.1,
   'subsample' : 0.8,
   'colsample_bytree' : 0.8,
   'gamma' : 0,
   'reg_alpha' : 0,
   'reg_lambda' : 1,
   'objective' : 'binary:logistic',
   'eval_metric' : 'logloss',
   'random_state' : 42,
   'number_files' : 10
}


@PipelineDecorator.pipeline(name='TPS_name_project', project='TPS_project', version='0.0.2')
def executing_pipeline(parameters):
   from clearml import Task, Logger

   # Инициализируйте объект Task
   task = Task.init(project_name="TPS_project/.pipelines/TPS_name_project", task_name="TPS_name_project # 16")


   parameters = task.connect(parameters)
   #logger = task.get_logger()

   path, labels, subs, sub_ids = step_one()
   ensemble_data = step_two(path, subs, parameters)
   X_train, X_val, y_train, y_val = step_three(ensemble_data, labels)
   y_pred_proba_val, y_pred_proba_test = step_four(X_train, X_val, y_train, ensemble_data, parameters)

   source = step_five(y_val, y_pred_proba_val)
   Logger.current_logger().report_scalar(title='accuracy', series='LogLoss', value=source, iteration=1)

   blend = step_six(y_pred_proba_test, sub_ids)
   blend.to_csv('blend.csv')
   Logger.current_logger().report_table("table blend", "PD with index", table_plot=blend)


if __name__ == '__main__':
   PipelineDecorator.run_locally()
   executing_pipeline(parameters)
   print('pipeline completed')