# pull official base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y

#RUN pip install --upgrade pip
RUN pip install pip-tools
COPY ./requirements.txt /app/requirements.txt

RUN pip-sync -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app/

ENV PYTHONPATH /app
ENV PORT 8000
ENV MAX_WORKERS 1

#CMD gunicorn -b 0.0.0.0 -w 1 -k uvicorn.workers.UvicornWorker app:app