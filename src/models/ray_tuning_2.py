from lightning.pytorch import LightningModule, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from src.services.timeseries_data_module import TimeSeriesDataModule

from lightning.pytorch.loggers import TensorBoardLogger



#context = ray.init()
#print(context.dashboard_url)

# Cria o logger e define a pasta de saída
tb_logger = TensorBoardLogger(
    save_dir="/tmp/ray/session_2025-12-27_15-10-45_546560_1483/artifacts",
    name="ray_train_run-2025-12-27_15-10-47"
)

class SimpleModel(LightningModule):
    def __init__(self, input_dim=60, hidden_dim=32, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        y_hat = self(X)
        loss = F.mse_loss(y_hat.squeeze(), y.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_func():
    data_module = TimeSeriesDataModule(
        csv_path="/app/data/raw/NVDA/2025-12-21.csv",
        value_col="Close",
        window_size=60,
        batch_size=2
    )
    data_module.setup()
    model = SimpleModel(input_dim=60, hidden_dim=32, output_dim=1)

    trainer = Trainer(
        logger=tb_logger,
        max_epochs=5,
        accelerator="gpu",
        devices=1 
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    ray.init(num_cpus=15)
    #ray.init(address="auto")   # conecta ao cluster existente
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()
    print("Fim")
    ray.shutdown()