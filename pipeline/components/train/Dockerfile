FROM python:3.7-slim-buster

WORKDIR /app

LABEL maintainer="ihuomacbasil@gmail.com"

# cache busting
RUN apt-get update && apt-get install -y

RUN pip install pandas

RUN pip install tensorflow 

# COPY more transparent than ADD
COPY train.py ./train.py

COPY ./src /pipelines/component/src

ENTRYPOINT [ "python", "train.py" ]