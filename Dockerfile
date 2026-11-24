# CPU-only image for serving the Streamlit app.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU wheels of torch keep the image ~1 GB instead of ~5 GB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY app.py .
# Put trained weights at models/pneumonia_resnet18.pt before building, or let the
# app download them from the latest GitHub Release at startup.
COPY models/ models/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
