# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for some packages (if needed)
RUN apt-get update && apt-get install -y build-essential git gcc && rm -rf /var/lib/apt/lists/*

# copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# use gunicorn for production-like server
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "--workers", "1", "--threads", "4"]
