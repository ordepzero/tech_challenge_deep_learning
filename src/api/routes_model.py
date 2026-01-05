from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.schemas.response import APIResponse
from src.schemas.train_request import TrainRequest
from src.schemas.prediction_request import PredictionRequest
from src.services.pipeline_service import train_job
from src.services.mlflow_service import MLFlowManager
from src.services.tuning_service import tune_model
from src.services.optimization_service import OptimizationService
from src.registry.task_registry import TaskRegistry
import ray
import torch
import logging
from datetime import datetime

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/models", tags=["models"])
registry = None # Will be initialized on startup

# Global cache for loaded models (simple in-memory storage)
LOADED_MODELS = {}

def get_registry():
    global registry
    if registry is None:
        try:
            registry = ray.get_actor("task_registry")
        except ValueError:
            # Create a detached actor so it persists even if the FastAPI app restarts
            registry = TaskRegistry.options(name="task_registry", lifetime="detached").remote()
    return registry

@router.get("/")
def list_models():
    """List all models/runs tracked by MLflow."""
    manager = MLFlowManager()
    runs = manager.list_runs()
    return APIResponse(status="success", message="Modelos listados do MLflow", data=runs)

@router.get("/{run_id}")
def get_model(run_id: str):
    """Get details of a specific model/run."""
    manager = MLFlowManager()
    try:
        details = manager.get_run_details(run_id)
        return APIResponse(status="success", message="Detalhes do modelo", data=details)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Modelo/Run não encontrado: {e}")

@router.post("/train")
def train_new_models(request: TrainRequest):
    """Start a new training job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Safe access to value if it's an enum, otherwise use string
    model_type_str = getattr(request.model_type, "value", str(request.model_type))
    
    model_id = f"{model_type_str}_{timestamp}"
    task_id = f"train_{model_id}"

    reg = get_registry()
    reg.set.remote(task_id, "running")
    train_job.remote(request, task_id, reg)

    return APIResponse(
        status="success",
        message="Treinamento iniciado",
        data={"model_id": model_id, "task_id": task_id}
    )

@router.post("/tune")
def tune_models(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start hyperparameter tuning."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type_str = getattr(request.model_type, "value", str(request.model_type))
    
    task_id = f"tune_{model_type_str}_{timestamp}"
    reg = get_registry()
    reg.set.remote(task_id, "running")

    def tuning_wrapper(req, t_id, registry_actor):
        try:
            tune_model(req)
            ray.get(registry_actor.set.remote(t_id, "completed"))
        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            ray.get(registry_actor.set.remote(t_id, f"failed: {e}"))

    # Running tuning in background to avoid blocking API
    background_tasks.add_task(tuning_wrapper, request, task_id, reg)
    return APIResponse(status="success", message="Otimização de hiperparâmetros iniciada em background.", data={"task_id": task_id})

@router.post("/{run_id}/load")
def load_model(run_id: str):
    """Load a model from MLflow into memory for prediction."""
    manager = MLFlowManager()
    try:
        model = manager.load_model(run_id)
        model.eval() # Set to eval mode
        LOADED_MODELS[run_id] = model
        logger.info(f"Model {run_id} loaded. Current keys: {list(LOADED_MODELS.keys())}")
        return APIResponse(status="success", message=f"Modelo {run_id} carregado na memória.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {e}")

@router.post("/predict")
def predict(request: PredictionRequest):
    """Make a prediction using a loaded model."""
    run_id = request.model_run_id
    logger.info(f"Predict request for {run_id}. Available models: {list(LOADED_MODELS.keys())}")
    
    if run_id not in LOADED_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo não carregado. Chame /load primeiro. Disponíveis: {list(LOADED_MODELS.keys())}")
    
    model = LOADED_MODELS[run_id]
    input_tensor = torch.tensor([request.data]) # Add batch dim
    
    # Check dimensions (simplified)
    # Assumes input is [seq_len, features] or similar depending on model
    
    try:
        with torch.no_grad():
            output = model(input_tensor)
        return APIResponse(status="success", message="Predição realizada", data=output.tolist())
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

@router.post("/{run_id}/prune")
def prune_model_endpoint(run_id: str, amount: float = 0.2):
    """Prune a specific model."""
    service = OptimizationService()
    try:
        result = service.prune_model(run_id, amount)
        return APIResponse(status="success", message=result)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erro ao aplicar pruning: {e}")

@router.post("/{run_id}/specialize")
def specialize_model_endpoint(run_id: str, request: TrainRequest):
    """Specialize (fine-tune) a model."""
    service = OptimizationService()
    task_id = f"specialize_{run_id}"
    reg = get_registry()
    reg.set.remote(task_id, "running")
    
    service.specialize_model(run_id, request, task_id, reg)
    return APIResponse(status="success", message="Especialização iniciada", data={"task_id": task_id})

@router.get("/status/{task_id}")
def get_status(task_id: str):
    reg = get_registry()
    state = ray.get(reg.get.remote(task_id))
    return APIResponse(status="success", message="Status da tarefa", data=state)
@router.get("/list_tasks")
def list_tasks():
    reg = get_registry()
    tasks = ray.get(reg.list.remote())
    return APIResponse(status="success", message="Lista de tarefas", data=tasks)
