FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install ollama pandas numpy

# Copy your RLM implementation
COPY main.py .

# Keep the container running or execute the script
CMD ["python", "main.py"]
