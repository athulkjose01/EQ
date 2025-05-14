FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add system dependencies
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    pkg-config \
    build-essential \
    libpq-dev

# Set work directory
WORKDIR /usr/src/app

# Install dependencies
COPY requirement.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirement.txt

# Copy project
COPY . /usr/src/app/

# Expose the port your app runs on
EXPOSE 8000

# Run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]