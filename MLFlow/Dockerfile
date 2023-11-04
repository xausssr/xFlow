FROM ubuntu:20.04

ENV TZ="UTC"
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir mlflow
RUN apt update && apt install -y
RUN apt install sqlite3 libsqlite3-dev -y
RUN apt install python3-pip -y && pip install pysqlite3
RUN apt install libpq-dev -y && apt install gcc -y

COPY ./requerments.txt /requerments.txt
RUN pip3 install -r /requerments.txt

RUN apt install postgresql postgresql-contrib postgresql-server-dev-all -y
COPY ./init_mlflow.sh ./init_mlflow.sh
COPY ./postgres_patch.conf ./postgres_patch.conf
RUN mkdir scripts
RUN mkdir artifacts

EXPOSE 5000

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
