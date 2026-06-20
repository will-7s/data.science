FROM python:3.12-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV DASH_REQUESTS_PATHNAME_PREFIX="/"

CMD ["gunicorn", "wsgi:server", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--preload", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
