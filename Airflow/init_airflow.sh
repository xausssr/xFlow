cp /inits/config_postgres /etc/postgresql/14/main/pg_hba.conf
pg_ctlcluster 14 main start
psql -U postgres -f /inits/init_airflow_postgres

airflow db init
cp /inits/airflow.cfg /root/airflow/airflow.cfg
airflow db init
airflow users create --username Admin --firstname admin --lastname test --role Admin --email root@root.local
