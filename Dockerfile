# Use official Python base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "lung_disease_api:app", "--host", "0.0.0.0", "--port", "8000"]
