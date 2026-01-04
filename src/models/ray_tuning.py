import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.services.data_loader import StockDataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.services.timeseries_data_module import RayTimeSeriesDataModule
from src.models.lstm_model import TimeSeriesModel

import ray
from ray.data import DataContext
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.lightning



def train_func():
    model = TimeSeriesModel(hidden_dim=64, num_layers=2, dropout_prob=0.2)
    model = ray.train.torch.prepare_model(model)

    # Treinamento
    trainer = L.Trainer(max_epochs=200, 
                        accelerator="auto", 
                        devices=1,
                        strategy=ray.train.lightning.RayDDPStrategy(),
                        plugins=[ray.train.lightning.RayLightningEnvironment()],
                        callbacks=[ray.train.lightning.RayTrainReportCallback()],
                        enable_checkpointing=False,
                        )
    trainer.fit(model, datamodule=data_module)




ray.init()
DataContext.get_current().enable_autoscaling = False  # desativa autoscaling do Ray Data

# Otimização para GPUs NVIDIA (RTX)
torch.set_float32_matmul_precision('medium')

# Configurar o loader para encontrar o arquivo
base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
loader = StockDataLoader(raw_path=f"{base_path}/data/raw", 
                            processed_path=f"{base_path}/data/processed")

# Obter o caminho do arquivo mais recente dinamicamente
filename_path = loader.get_latest_file_path("NVDA", kind="raw")
print(f"Utilizando arquivo: {filename_path}")


data_module = RayTimeSeriesDataModule(
    csv_path=filename_path,
    value_col="Close",
    window_size=5
)

ds_train, ds_val, ds_test = data_module.get_ray_datasets()

# Iterando com conversão para tensores
it = ds_train.iter_batches(batch_size=2)
try:
    for i, batch in enumerate(it):
        # conversão para tensores
        x = torch.tensor(np.stack(batch["x"]), dtype=torch.float32)
        y = torch.tensor(batch["y"], dtype=torch.float32)
        base = torch.tensor(batch["base"], dtype=torch.float32)
        print(x.shape, y.shape, base.shape)
        if i == 0:  # consome só um batch para teste
            break
finally:
    # encerra o iterador e libera recursos do executor
    if hasattr(it, "close"):
        it.close()


### Ray Train ###
    
# Multiple workers, each with a GPU
scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()



# encerra o Ray de forma explícita após terminar tudo
ray.shutdown()