from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

import ray.train.lightning
from ray.train.torch import TorchTrainer
import torch
import ray
from ray.train.lightning import RayDDPStrategy

from src.schemas.train_request import TrainRequest 
from src.services.data_loader import StockDataLoader
from src.services.timeseries_data_module import TimeSeriesDataModule
from src.models.factory import ModelFactory
from src.registry.task_registry import TaskRegistry
import logging
import mlflow

# Configuração base do logger para trabalhos de treinamento
logger = logging.getLogger("train_job")

def configure_logging(level_name: str):
    """
    Configura dinamicamente o nível de log para a execução atual.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level)
    logger.setLevel(level)
    # Ajusta os loggers das bibliotecas dependentes para manter consistência
    logging.getLogger("ray").setLevel(level)
    logging.getLogger("lightning").setLevel(level)

@ray.remote(num_gpus=1)
def train_job(config: TrainRequest, task_id: str, registry: TaskRegistry):
    """
    Função remota do Ray para orquestrar um trabalho de treinamento completo.
    """
    try:
        configure_logging(config.log_level)
        logger.info(f"Iniciando tarefa {task_id} com nível de log {config.log_level}")
        
        run_train(config, task_id)
        
        # Atualiza o status no TaskRegistry como concluído
        ray.get(registry.set.remote(task_id, "completed"))
        logger.info(f"Tarefa {task_id} concluída com sucesso.")
    except Exception as e:
        logger.error(f"Tarefa {task_id} falhou: {e}", exc_info=True)
        # Registra a falha no TaskRegistry
        ray.get(registry.set.remote(task_id, f"failed: {e}"))


def run_train(config: TrainRequest, task_id: str):
    """
    Lógica principal de treinamento utilizando PyTorch Lightning e integração com MLflow.
    """
    logger.info("Dentro da função de treinamento, GPUs detectadas no nó: %s", torch.cuda.device_count())

    # Inicializa a configuração do MLflow com banco de dados SQLite para visibilidade na UI
    import os
    if os.path.exists("/app"):
        tracking_uri = "sqlite:////app/mlflow.db"
    else:
        tracking_uri = "sqlite:///mlflow.db"
        
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("stock_price_prediction")
    
    # Configura o Logger do Lightning para o MLflow
    mlflow_logger = MLFlowLogger(experiment_name="stock_price_prediction", run_name=task_id, tracking_uri=tracking_uri)
    
    # Log dos hiperparâmetros iniciais
    mlflow_logger.log_hyperparams(config.dict())

    base_path = "." # Assume execução no diretório raiz do projeto
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw",
                             processed_path=f"{base_path}/data/processed")

    # Obtém o arquivo de dados mais recente (atualmente fixado para NVDA)
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    logger.info(f"Usando arquivo de dados: {filename_path}")
    
    # Registra o caminho do dataset como parâmetro no MLflow
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "dataset_path", filename_path)

    # Inicializa o DataModule para séries temporais
    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",
        window_size=config.window_size,
        batch_size=32
    )
    data_module.setup()
    
    # Cria o modelo através da factory com base na configuração
    model = ModelFactory.create(config)

    # Configuração de callbacks: Checkpoint e Early Stopping
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='lstm-best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=(config.log_level == "DEBUG"),
        mode='min'
    )

    # Estratégia de treinamento: Decide entre treinamento local ou distribuído com RayDDP
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs disponíveis para a estratégia: {num_gpus}")

    if num_gpus > 1:
        logger.info("Usando RayDDPStrategy para treinamento distribuído...")
        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            devices=num_gpus,
            strategy=RayDDPStrategy(),
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=mlflow_logger,
            enable_progress_bar=(config.log_level == "DEBUG")
        )
        # Prepara o trainer para integração com o ecossistema Ray Train
        trainer = ray.train.lightning.prepare_trainer(trainer)
    else:
        logger.info("Usando treinamento local (CPU ou 1 GPU)...")
        devices = 1 if num_gpus > 0 else "auto"
        accelerator = "gpu" if num_gpus > 0 else "cpu"
        
        trainer = Trainer(
            max_epochs=200,
            accelerator=accelerator,
            devices=devices,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=mlflow_logger,
            enable_progress_bar=(config.log_level == "DEBUG")
        )

    # Início do ciclo de treinamento
    logger.info("Iniciando loop de treinamento...")
    trainer.fit(model, datamodule=data_module)

    # Execução de testes com o melhor modelo encontrado
    logger.info("Iniciando fase de testes com o melhor modelo...")   
    trainer.test(model, datamodule=data_module)
    
    # Salvamento e registro do melhor modelo no MLflow
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Caminho do melhor modelo: {best_model_path}")
        # Loga o arquivo de checkpoint como artefato
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, best_model_path, artifact_path="best_checkpoint")
        
        # Registra o modelo no formato nativo do MLflow (PyTorch)
        best_model = ModelFactory.create(config) # Re-instancia a arquitetura
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['state_dict'])
        
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.pytorch.log_model(best_model, artifact_path="model")

    logger.info("Treinamento finalizado com sucesso!")

