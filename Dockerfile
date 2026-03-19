FROM python:3.12-slim

# Set working directory
WORKDIR /app
COPY . /app
# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "boucle0.py"]
