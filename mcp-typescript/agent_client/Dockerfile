FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "300", "-b", "0.0.0.0:8000", "app:app"]
