FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

RUN mkdir scripts

RUN apt update -y
RUN apt install software-properties-common -y
RUN apt install python-setuptools -y
RUN apt install libpq-dev -y
RUN apt install postgresql -y
RUN export AIRFLOW_HOME=~/airflow


COPY ./requerments.txt /requerments.txt

RUN pip3 install -r /requerments.txt

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]