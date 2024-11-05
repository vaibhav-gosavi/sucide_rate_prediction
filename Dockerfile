# Use an official Python image as a base
FROM python:3.11-bullseye

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker caching for dependencies
COPY requirements_D.txt ./requirements.txt

# Install the dependencies - this step will be cached if requirements.txt hasn't changed
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the command to start the development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
