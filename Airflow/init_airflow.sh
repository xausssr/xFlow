airflow db init
airflow users create --username Admin --firstname admin --lastname test --role Admin --email root@root.local

mkdir /root/airflow/dags

airflow webserver -p 8080 -D
airflow scheduler -D