from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
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
logger = logging.getLogger("train_job")
logger.setLevel(logging.INFO)

@ray.remote
def train_job(config: TrainRequest, task_id: str, registry: TaskRegistry):
    try:
        # model = ModelFactory.create(config)
        # roda o treino normalmente
        run_train(config)
        ray.get(registry.set.remote(task_id, "completed"))
    except Exception as e:
        ray.get(registry.set.remote(task_id, f"failed: {e}"))


@ray.remote(num_gpus=1)
def run_train(config: TrainRequest):
    logger.info("Dentro da task, GPUs detectadas: %s", torch.cuda.device_count())

    base_path = "."
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw",
                             processed_path=f"{base_path}/data/processed")

    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    print(f"Utilizando arquivo: {filename_path}")

    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",
        window_size=60,
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
        verbose=True,
        mode='min'
    )

    # --- Escolha automática da estratégia ---
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detectadas: {num_gpus}")
    return True

    if num_gpus > 1:
        print("Usando RayDDPStrategy para treino distribuído...")
        trainer = Trainer(
            max_epochs=200,
            accelerator="gpu",
            devices=num_gpus,
            strategy=RayDDPStrategy(),
            callbacks=[checkpoint_callback, early_stop_callback]
        )
        trainer = ray.train.lightning.prepare_trainer(trainer)
    else:
        print("Usando treino local (CPU ou 1 GPU)...")
        trainer = Trainer(
            max_epochs=200,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback, early_stop_callback]
        )

    # --- Treinamento ---
    print("\nIniciando treinamento...")
    trainer.fit(model, datamodule=data_module)

    print("\nIniciando Teste com o melhor modelo...")   
    trainer.test(model, datamodule=data_module)
    print("\nTreinamento finalizado com sucesso!")