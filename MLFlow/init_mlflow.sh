cp ./postgres_patch.conf /etc/postgresql/12/main/pg_hba.conf

pg_ctlcluster 12 main start
psql -U postgres -c "CREATE DATABASE mlflow;"
psql -U postgres -c "CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;"

nohup mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root file:/artifacts -h 0.0.0.0 -p 5000 &
