from fastapi import APIRouter, HTTPException
from src.schemas.response import APIResponse
from src.schemas.train_request import TrainRequest
from src.services.pipeline_service import train_job
from src.registry.task_registry import TaskRegistry
import ray
import logging
logger = logging.getLogger("uvicorn")


router = APIRouter(prefix="/models", tags=["models"])
registry = TaskRegistry.remote()


# Lista de modelos simulada
MODELS = {"bert_v1": {"status": "ready"}, "xgboost_v2": {"status": "trained"}}
TASKS = {}




@router.get("/")
def list_models():
    return APIResponse(status="success", message="Modelos disponíveis", data=MODELS)

@router.get("/{model_id}")
def get_model(model_id: str):
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    return APIResponse(status="success", message="Modelo encontrado", data=MODELS[model_id])

@router.post("/{model_id}/train")
def retrain_model(model_id: str):
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    task_id = f"train_{model_id}"
    TASKS[task_id] = {"state": "running"}
    return APIResponse(status="success", message="Treinamento iniciado", data={"task_id": task_id})

@router.post("/train")
def train_new_models(request: TrainRequest):
    model_id = f"{request.model_type}_1"
    task_id = f"train_{model_id}"

    registry.set.remote(task_id, "running")
    train_job.remote(request, task_id, registry)

    return APIResponse(
        status="success",
        message="Treinamento iniciado",
        data={"model_id": model_id, "task_id": task_id}
    )

@router.get("/status/{task_id}")
def get_status(task_id: str):
    state = ray.get(registry.get.remote(task_id))
    return APIResponse(status="success", message="Status da tarefa", data={"task_id": task_id, "state": state})

@router.get("/list_tasks")
def list_tasks():
    tasks = ray.get(registry.list.remote())
    return APIResponse(status="success", message="Lista de tarefas", data=tasks)
