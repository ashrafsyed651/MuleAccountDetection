FROM python:3.9-slim

WORKDIR /feature_store
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
