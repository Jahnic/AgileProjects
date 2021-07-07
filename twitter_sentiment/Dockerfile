FROM python:3.8

COPY src /app/src
COPY requirements.txt /app
COPY setup.py /app

RUN mkdir /app/models
# need -p since parent directory /data needs to be created before /raw
RUN mkdir -p /app/data/raw

COPY models/binary_transformer_model.pt /app/models
COPY data/raw/twitter_creds.json /app/data/raw

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

WORKDIR /app/src

CMD streamlit run --server.port 8080 --server.enableCORS false app.py