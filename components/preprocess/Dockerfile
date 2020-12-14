FROM python:3.7-slim-buster

WORKDIR /app

LABEL maintainer="ihuomacbasil@gmail.com"

RUN pip install -U scikit-learn numpy

RUN pip install pandas

RUN pip install tensorflow

# cache busting
RUN apt-get update && apt-get install -y

# COPY more transparent than ADD
COPY preprocess.py ./preprocess.py

COPY ./src /pipelines/component/src

ENTRYPOINT [ "python", "preprocess.py" ]