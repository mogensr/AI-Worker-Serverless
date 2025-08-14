# Step 1: Start with a specific, stable Python version
FROM python:3.10-slim

# Step 2: Install system-level dependencies, including Git and build tools
RUN apt-get update && apt-get install -y git build-essential

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Step 5: Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Expose the port that FastAPI will run on
EXPOSE 8000

# Step 8: Define the command to run your application
# Uvicorn is a high-performance server for FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
