FROM rayproject/ray:latest-py312-gpu

WORKDIR /app

COPY requirements.txt .
USER root

# Install stable PyTorch with CUDA 12.1 support (compatible with Ray stable)
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install supervisor uvicorn

COPY . .

EXPOSE 8265 10001 8000

COPY supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf"]