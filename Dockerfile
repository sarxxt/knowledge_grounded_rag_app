FROM python:3.8-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

# Install required tools and update keyring
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    gnupg2 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138 \
    && apt-get install -y poppler-utils tesseract-ocr

RUN pip install "unstructured[all-docs]"
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
