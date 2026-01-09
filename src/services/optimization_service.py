import mlflow
import torch
import torch.nn.utils.prune as prune
from src.services.mlflow_service import MLFlowManager
from src.services.pipeline_service import train_job, run_train
from src.schemas.train_request import TrainRequest
from src.registry.task_registry import TaskRegistry
import ray
import logging

# Configuração do logger local
logger = logging.getLogger(__name__)

class OptimizationService:
    """
    Serviço responsável por otimizações de modelos, como pruning (poda) e especialização (fine-tuning).
    """
    def __init__(self):
        """
        Inicializa o OptimizationService com uma instância do MLFlowManager.
        """
        self.mlflow_manager = MLFlowManager()

    def prune_model(self, run_id: str, amount: float = 0.2):
        """
        Aplica o pruning L1 não estruturado às camadas lineares do modelo identificado pelo run_id.
        """
        logger.info(f"Aplicando pruning ao modelo da run {run_id} com quantidade {amount}")
        
        # Carrega o modelo do MLflow
        model = self.mlflow_manager.load_model(run_id)
        
        # Aplica pruning a todas as camadas lineares (estratégia de exemplo)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight') # Torna o pruning permanente
        
        # Registra o modelo podado de volta no MLflow como uma nova run
        new_run_id = None
        with mlflow.start_run(run_name=f"pruned_{run_id}") as run:
            new_run_id = run.info.run_id
            mlflow.log_param("original_run_id", run_id)
            mlflow.log_param("pruning_amount", amount)
            mlflow.pytorch.log_model(model, "model")
            
        logger.info(f"Pruning concluído. Nova run_id: {new_run_id}")
        return {"status": "Pruning completed", "new_run_id": new_run_id}

    def specialize_model(self, original_run_id: str, train_request: TrainRequest, task_id: str, registry):
        """
        Realiza o fine-tuning de um modelo pré-treinado em novos dados ou parâmetros.
        Nota: Atualmente redireciona para o pipeline de treinamento padrão como um protótipo.
        """
        logger.info(f"Especializando modelo a partir da run {original_run_id}")
        
        # Estratégia atual: Dispara um trabalho de treinamento padrão.
        # Em uma implementação robusta, o modelo inicial seria carregado do original_run_id.
        
        # Dispara o trabalho via Ray
        train_job.remote(train_request, task_id, registry)
        
        return f"Tarefa de especialização {task_id} iniciada."

