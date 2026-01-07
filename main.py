from contextlib import asynccontextmanager
import logging
import ray
import torch
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.api import routes_model, routes_data

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if torch.cuda.is_available():
        logger.info(f"CUDA is available! Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA is NOT available. Ray might not detect GPUs.")

    try:
        # Initialize Ray locally since we are in a single container
        # address=None forces a local cluster to be started
        ray.init(
            address=None,
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            _metrics_export_port=8080,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        logger.info("Ray initialized successfully (Local Cluster).")
        logger.info(f"Ray Resources: {ray.cluster_resources()}")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")

    yield
    
    # Shutdown
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown.")

app = FastAPI(lifespan=lifespan)

# Instrument FastAPI
Instrumentator().instrument(app).expose(app)

app.include_router(routes_model.router)
app.include_router(routes_data.router)

@app.get("/")
async def root():
    return {"message": "Hello World", "ray_status": "active" if ray.is_initialized() else "inactive"}