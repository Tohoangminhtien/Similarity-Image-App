FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install pandas Flask tqdm requests transformers opencv-python tensorflow pymilvus tf-keras Pillow


COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "application.py"]
