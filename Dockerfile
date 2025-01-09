FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY emotion_model.onnx .
COPY emotion_model-8classes.onnx .

COPY static static
COPY templates templates
COPY app.py .

CMD ["python3", "app.py"]

EXPOSE 8888