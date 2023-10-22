FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

RUN mkdir scripts

ENV TZ="UTC"

RUN apt update -y
RUN apt install software-properties-common -y
RUN apt install python-setuptools -y
RUN apt install libpq-dev -y
RUN export AIRFLOW_HOME="/root/airflow/"
RUN export AIRFLOW__CORE__LOAD_EXAMPLES=False


COPY ./requerments.txt /requerments.txt

RUN pip3 install -r /requerments.txt

#COPY inits /inits
COPY init_airflow.sh /init_airflow.sh

EXPOSE 8080

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]