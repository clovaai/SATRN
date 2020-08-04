FROM tensorflow/tensorflow:1.12.0-gpu-py3

COPY . /app

RUN apt update && apt install -y libsm6 libxext6 libxrender1

WORKDIR /app
RUN pip install -r requirements.txt

CMD ["bash"]
