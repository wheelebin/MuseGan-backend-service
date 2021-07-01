# pull official base image
FROM tiangolo/uvicorn-gunicorn:python3.6

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

#RUN pip install --upgrade pip
RUN pip install pip-tools
COPY ./requirements.txt /app/requirements.txt

RUN pip-sync -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app/

