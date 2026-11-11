FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
RUN apt-get -o Acquire::Retries=4 update && apt-get -o Acquire::Retries=4 install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 90 -r requirements-serving.txt

COPY src/ ./src/
COPY deployment/app.py ./deployment/app.py
COPY models/best_model.pkl ./models/best_model.pkl

USER 10001
EXPOSE 8000
CMD ["python", "-m", "deployment.app"]
