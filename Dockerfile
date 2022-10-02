# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
COPY . /app
# RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r /app/requirements.txt

CMD streamlit run /app/streamlit_app.py
