FROM rayproject/ray:nightly-py312-gpu

WORKDIR /app

COPY requirements.txt .
USER root

RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install supervisor uvicorn

COPY . .

EXPOSE 8265 10001 8000

COPY supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf"]