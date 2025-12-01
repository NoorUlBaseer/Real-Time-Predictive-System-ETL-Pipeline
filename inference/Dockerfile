FROM python:3.9-slim

WORKDIR /app

# Install necessary Python packages
RUN pip install --no-cache-dir scikit-learn pandas fastapi uvicorn joblib

# Copy application code and model directory
COPY app.py .
COPY model_dir/ ./model_dir/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]