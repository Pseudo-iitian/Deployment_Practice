# Use the official Python image for version 3.10.12 from the Docker Hub
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run the app
CMD ["python3", "app.py"]

# Expose the port the app runs on
EXPOSE 5000
