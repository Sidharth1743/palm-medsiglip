FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/palm_deploy
COPY artifacts /app/artifacts

ENV ARTIFACT_DIR=/app/artifacts

EXPOSE 8080
CMD ["uvicorn", "palm_deploy.app:app", "--host", "0.0.0.0", "--port", "8080"]
