# فایل جدید — استقرار سرویس پیش‌بینی ریزش مشتری (مرحله ۴ مستند پروژه)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/best_model.pkl ./models/best_model.pkl
COPY deployment/app.py ./deployment/app.py

ENV MODEL_PATH=/app/models/best_model.pkl
EXPOSE 8000

CMD ["python", "deployment/app.py"]
