# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pydantic-settings requests streamlit

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run both processes (in a real production we'd use a supervisor or separate containers)
CMD python scripts/initialize_db.py && \
    uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
