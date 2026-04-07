# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install mlflow dagshub gunicorn

# Copy all project files
COPY . .

# Expose port
EXPOSE 5000

# Environment variables (for Flask)
ENV PYTHONUNBUFFERED=1

# Run app using gunicorn (production server)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]