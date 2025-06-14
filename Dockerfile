# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code from the host to the image's filesystem at /app
COPY *.py ./

# Run app.py when the container launches
ENTRYPOINT streamlit run st_app.py --server.port=$PORT --server.address=0.0.0.0
