FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install ollama pandas numpy datasets

# Copy your RLM implementation
COPY boucle0.py .

# Keep the container running or execute the script
CMD ["tail", "-f", "/dev/null"]