# FROM python:3.7-slim-buster
FROM google/cloud-sdk:latest

WORKDIR /app

LABEL maintainer="ihuomacbasil@gmail.com"

RUN chmod 755 /pipelines/component/src/deploy.sh

# RUN pip install -U scikit-learn numpy

# RUN pip install pandas

# RUN pip install tensorflow

# cache busting
# RUN apt-get update && apt-get install -y

# COPY more transparent than ADD
COPY deploy_model.py ./deploy_model.py

COPY ./src /pipelines/component/src

ENTRYPOINT ["/pipelines/component/src/deploy.sh"]

# ENTRYPOINT [ "python", "deploy.sh" ]







