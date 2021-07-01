# pull official base image
FROM python:3.6-slim-buster

# set work directory
WORKDIR /usr/src/app

# set environment variables
#ENV PYTHONDONTWRITEBYTECODE 1

# install dependencies
RUN pip install --upgrade pip
RUN pip install pip-tools
COPY ./requirements.txt /usr/src/app/requirements.txt
#RUN pip install -r requirements.txt
RUN pip-sync

# copy project
COPY . /usr/src/app/