from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy_operator import DummyOperator
import datetime as dt

args = {
    "owner": "airflow",
    "start_date": dt.datetime(2022, 12, 1),
}

dag = DAG(dag_id="HH_salary", default_args=args, schedule_interval="@once", catchup=False)

get_data = BashOperator(task_id="get_data", bash_command="python /scripts/airflow/get_data.py", dag=dag)
build_dataset = BashOperator(task_id="build_dataset", bash_command="python /scripts/airflow/build_dataset.py", dag=dag)
train_model = BashOperator(task_id="train_model", bash_command="python /scripts/airflow/train_model.py", dag=dag)
evaluate_model = BashOperator(
    task_id="evaluate_model", bash_command="python /scripts/airflow/evaluate_model.py", dag=dag
)

get_data >> build_dataset >> train_model >> evaluate_model
