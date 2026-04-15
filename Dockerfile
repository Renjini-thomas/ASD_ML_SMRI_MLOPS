FROM python:3.10-slim

WORKDIR /app

# 🔥 ADD THIS BLOCK (VERY IMPORTANT)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install mlflow dagshub gunicorn

# Copy project
COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]