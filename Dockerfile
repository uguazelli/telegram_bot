FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start both services using a process manager
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & python telegram_handler.py"]
