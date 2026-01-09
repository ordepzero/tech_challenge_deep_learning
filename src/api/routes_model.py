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

from src.services.data_loader import StockDataLoader
import os

# Configuração do logger uvicorn
logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/models", tags=["models"])
registry = None # Será inicializado no startup

# Define o caminho base para os dados (suporta local e container)
BASE_DATA_PATH = "." # Ou /app no container
LOADER = StockDataLoader(
    raw_path=f"{BASE_DATA_PATH}/data/raw",
    processed_path=f"{BASE_DATA_PATH}/data/processed"
)

# Cache global para modelos carregados (armazenamento simples em memória)
LOADED_MODELS = {}

def get_registry():
    """
    Obtém ou inicializa o TaskRegistry como um ator do Ray.
    Utiliza um ator 'detached' para que persista mesmo que o app FastAPI reinicie.
    """
    global registry
    if registry is None:
        try:
            registry = ray.get_actor("task_registry")
        except ValueError:
            # Cria um ator desacoplado para persistência
            registry = TaskRegistry.options(name="task_registry", lifetime="detached").remote()
    return registry

@router.get("/")
def list_models():
    """
    Lista todos os modelos e execuções (runs) rastreados pelo MLflow.
    """
    manager = MLFlowManager()
    runs = manager.list_runs()
    return APIResponse(status="success", message="Modelos listados do MLflow", data=runs)

@router.get("/list_tasks")
def list_tasks():
    """
    Lista todas as tarefas de background registradas no TaskRegistry.
    """
    reg = get_registry()
    tasks = ray.get(reg.list.remote())
    return APIResponse(status="success", message="Lista de tarefas", data=tasks)

@router.get("/status/{task_id}")
def get_status(task_id: str):
    """
    Obtém o status de uma tarefa específica via task_id.
    """
    reg = get_registry()
    state = ray.get(reg.get.remote(task_id))
    return APIResponse(status="success", message="Status da tarefa", data=state)

@router.get("/{run_id}")
def get_model(run_id: str):
    """
    Obtém os detalhes de um modelo/run específico no MLflow.
    """
    manager = MLFlowManager()
    try:
        details = manager.get_run_details(run_id)
        return APIResponse(status="success", message="Detalhes do modelo", data=details)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Modelo/Run não encontrado: {e}")

@router.post("/train")
def train_new_models(request: TrainRequest):
    """
    Inicia um novo trabalho de treinamento de modelo.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Acesso seguro ao valor se for um enum, caso contrário usa string
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
    """
    Inicia a otimização de hiperparâmetros (tuning).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type_str = getattr(request.model_type, "value", str(request.model_type))
    
    task_id = f"tune_{model_type_str}_{timestamp}"
    reg = get_registry()
    reg.set.remote(task_id, "running")

    def tuning_wrapper(req, t_id, registry_actor):
        """
        Wrapper para executar o tuning em background e atualizar o registro de tarefas.
        """
        try:
            result = tune_model(req)
            # result agora é um dicionário com {"config": ..., "run_id": ...}
            ray.get(registry_actor.set.remote(t_id, {"status": "completed", "run_id": result.get("run_id")}))
        except Exception as e:
            logger.error(f"Falha no tuning: {e}", exc_info=True)
            ray.get(registry_actor.set.remote(t_id, f"failed: {e}"))

    # Executa o tuning em background para não bloquear a API
    background_tasks.add_task(tuning_wrapper, request, task_id, reg)
    return APIResponse(status="success", message="Otimização de hiperparâmetros iniciada em background.", data={"task_id": task_id})

@router.post("/{run_id}/load")
def load_model(run_id: str):
    """
    Carrega um modelo do MLflow para a memória para fins de predição.
    """
    manager = MLFlowManager()
    try:
        model = manager.load_model(run_id)
        model.eval() # Define para o modo de avaliação
        LOADED_MODELS[run_id] = model
        logger.info(f"Modelo {run_id} carregado. Chaves atuais: {list(LOADED_MODELS.keys())}")
        return APIResponse(status="success", message=f"Modelo {run_id} carregado na memória.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {e}")

@router.post("/predict")
def predict(request: PredictionRequest):
    """
    Realiza uma predição com normalização automática e preenchimento de janela de dados.
    """
    run_id = request.model_run_id
    logger.info(f"Requisição de predição para {run_id}. Ticker: {request.ticker}")
    
    if run_id not in LOADED_MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo não carregado. ID solicitado: '{run_id}'. Disponíveis: {list(LOADED_MODELS.keys())}")
    
    model = LOADED_MODELS[run_id]
    expected_window = getattr(model, "hparams", {}).get("window_size", 60)
    
    input_data = request.data
    
    # 0. Preenchimento automático da janela se dados parciais forem fornecidos
    if len(input_data) < expected_window:
        if not request.ticker:
            raise HTTPException(
                status_code=400, 
                detail=f"Dados insuficientes ({len(input_data)}/{expected_window}). Informe o 'ticker' para completar a janela automaticamente."
            )
        
        try:
            # Tenta carregar histórico do CSV local
            df = LOADER.load_raw(request.ticker)
            # Pega o histórico mais recente (coluna Close)
            history = df['Close'].tail(expected_window).values.tolist()
            
            needed = expected_window - len(input_data)
            if len(history) < needed:
                 raise HTTPException(status_code=400, detail=f"Histórico insuficiente no arquivo para o ticker {request.ticker}. Encontrados {len(history)} registros.")
            
            # Concatena: [Histórico Antigo] + [Dados do Usuário]
            input_data = history[-needed:] + input_data
            logger.info(f"Janela completada para {request.ticker}. Novos dados: {len(input_data)} valores.")
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Arquivo histórico não encontrado para o ticker {request.ticker}. Por favor, realize o download primeiro.")
        except Exception as e:
            logger.error(f"Erro ao completar janela: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao completar janela historicamente: {e}")

    # Validação final de tamanho
    if len(input_data) != expected_window:
         raise HTTPException(status_code=400, detail=f"Tamanho da série final inválido. Esperado {expected_window}, obtido {len(input_data)}.")

    # 1. Normalização Automática (baseada no histórico fornecido)
    # De acordo com TimeSeriesDataset: base_value = x[-1]
    base_value = input_data[-1]
    if base_value == 0:
        base_value = 1e-8
        
    # Normaliza a janela: (valor / base) - 1
    data_norm = [(v / base_value) - 1 for v in input_data]
    
    input_tensor = torch.tensor([data_norm], dtype=torch.float32) # Adiciona dimensão de batch
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        # 2. Denormalização: Preço = (Norm + 1) * Base
        predicted_return = output.item() if output.numel() == 1 else output.tolist()
        
        if isinstance(predicted_return, (float, int)):
            predicted_price = (predicted_return + 1) * base_value
        else:
            predicted_price = [(r + 1) * base_value for r in predicted_return]

        return APIResponse(
            status="success", 
            message="Predição realizada com sucesso", 
            data={
                "predicted_price": predicted_price,
                "predicted_return": predicted_return,
                "base_value_used": base_value,
                "window_completed": len(request.data) < expected_window
            }
        )
    except Exception as e:
        logger.error(f"Erro durante a predição: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

@router.post("/{run_id}/prune")
def prune_model_endpoint(run_id: str, amount: float = 0.2):
    """
    Aplica pruning a um modelo específico.
    """
    service = OptimizationService()
    try:
        result = service.prune_model(run_id, amount)
        return APIResponse(status="success", message="Pruning completed", data=result)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Erro ao aplicar pruning: {e}")

@router.post("/{run_id}/specialize")
def specialize_model_endpoint(run_id: str, request: TrainRequest):
    """
    Inicia a especialização (fine-tuning) de um modelo.
    """
    service = OptimizationService()
    task_id = f"specialize_{run_id}"
    reg = get_registry()
    reg.set.remote(task_id, "running")
    
    service.specialize_model(run_id, request, task_id, reg)
    return APIResponse(status="success", message="Especialização iniciada", data={"task_id": task_id})

