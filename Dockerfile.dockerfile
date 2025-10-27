# Use Python 3.11 as the base image (matches your local Python version from the error trace)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first (for better caching during builds)
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (code, data, config, etc.)
COPY . .

# Set the entrypoint to run main.py
ENTRYPOINT ["python", "main.py"]