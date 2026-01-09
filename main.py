from contextlib import asynccontextmanager
import logging
import ray
import torch
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.api import routes_model, routes_data

# Configuração do logger para o uvicorn
logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação FastAPI.
    Realiza a inicialização do Ray e verifica a disponibilidade de GPU/CUDA.
    """
    # Inicialização (Startup)
    if torch.cuda.is_available():
        logger.info(f"CUDA está disponível! Dispositivos: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"Dispositivo {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA NÃO está disponível. O Ray pode não detectar GPUs.")

    try:
        # Inicializa o Ray localmente, já que estamos em um único container.
        # address=None força o início de um cluster local.
        ray.init(
            address=None,
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            _metrics_export_port=8080,
            num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        logger.info("Ray inicializado com sucesso (Cluster Local).")
        logger.info(f"Recursos do Ray: {ray.cluster_resources()}")
    except Exception as e:
        logger.error(f"Falha ao inicializar o Ray: {e}")

    yield
    
    # Finalização (Shutdown)
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray finalizado.")

# Inicializa a instância do FastAPI com o contexto de ciclo de vida definido
app = FastAPI(lifespan=lifespan)

# Instrumentação do FastAPI para coleta de métricas pelo Prometheus
Instrumentator().instrument(app).expose(app)

# Inclusão das rotas da API
app.include_router(routes_model.router)
app.include_router(routes_data.router)

@app.get("/")
async def root():
    """
    Rota raiz para verificação básica de saúde da API.
    Retorna o status de ativação do Ray.
    """
    return {"message": "Hello World", "ray_status": "active" if ray.is_initialized() else "inactive"}
