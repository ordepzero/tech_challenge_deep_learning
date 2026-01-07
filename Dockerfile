FROM rayproject/ray:latest-py312-gpu

WORKDIR /app

COPY requirements.txt .
USER root

# Install PyTorch Nightly (Required for RTX 50-series / sm_120 support)
RUN apt-get update && apt-get install -y socat && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install supervisor uvicorn

COPY . .

EXPOSE 8265 10001 8000 5000

COPY supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf"]