
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

# Configuração base do logger
logger = logging.getLogger("train_job")

def configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level)
    logger.setLevel(level)
    # Ajusta loggers do Ray e Lightning se necessário
    logging.getLogger("ray").setLevel(level)
    logging.getLogger("lightning").setLevel(level)

@ray.remote(num_gpus=1)
def train_job(config: TrainRequest, task_id: str, registry: TaskRegistry):
    try:
        configure_logging(config.log_level)
        logger.info(f"Starting task {task_id} with log level {config.log_level}")
        
        run_train(config, task_id)
        
        ray.get(registry.set.remote(task_id, "completed"))
        logger.info(f"Task {task_id} completed successfully.")
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        ray.get(registry.set.remote(task_id, f"failed: {e}"))


def run_train(config: TrainRequest, task_id: str):
    logger.info("Inside training function, detected GPUs on node: %s", torch.cuda.device_count())

    # Inicializa MLFlow Logger
    import os
    tracking_uri = "file:///app/mlruns" if os.path.exists("/app") else "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("stock_price_prediction")
    
    mlflow_logger = MLFlowLogger(experiment_name="stock_price_prediction", run_name=task_id, tracking_uri=tracking_uri)
    
    # Log dos parâmetros iniciais
    mlflow_logger.log_hyperparams(config.dict())

    base_path = "."
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw",
                             processed_path=f"{base_path}/data/processed")

    # TODO: Parametrizar ticker se necessário, por enquanto hardcoded como no original
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    logger.info(f"Using file: {filename_path}")
    
    # Log do dataset usado
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "dataset_path", filename_path)

    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",
        window_size=config.window_size,
        batch_size=32
    )
    data_module.setup()
    model = ModelFactory.create(config)

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

    # --- Escolha automática da estratégia ---
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs available for strategy: {num_gpus}")

    if num_gpus > 1:
        logger.info("Using RayDDPStrategy for distributed training...")
        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            devices=num_gpus,
            strategy=RayDDPStrategy(),
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=mlflow_logger,
            enable_progress_bar=(config.log_level == "DEBUG")
        )
        trainer = ray.train.lightning.prepare_trainer(trainer)
    else:
        logger.info("Using local training (CPU or 1 GPU)...")
        # Se tiver 1 GPU, usa. Se 0, usa CPU (auto)
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

    # --- Treinamento ---
    logger.info("Starting training loop...")
    trainer.fit(model, datamodule=data_module)

    logger.info("Starting testing with best model...")   
    trainer.test(model, datamodule=data_module)
    
    # Salvar o melhor modelo no MLFlow
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model path: {best_model_path}")
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, best_model_path, artifact_path="best_checkpoint")
        
        # Logar o modelo como artefato do MLFlow (PyTorch format)
        # Carrega o melhor modelo para salvar no formato padrão do MLflow
        best_model = ModelFactory.create(config) # Re-instanciar arquitetura
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['state_dict'])
        
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.pytorch.log_model(best_model, artifact_path="model")

    logger.info("Training finished successfully!")
