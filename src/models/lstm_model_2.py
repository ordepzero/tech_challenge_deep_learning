import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule, Trainer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.services.data_loader import StockDataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.services.timeseries_data_module import TimeSeriesDataModule


# Lightning Module for LSTM Model
class LSTMModel2(LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_prob: float = 0.2,
        learning_rate: float = 0.001,
        window_size: int = 60
    ):
        super().__init__()
        # Salva todos os hiperparâmetros automaticamente em self.hparams
        self.save_hyperparameters()

        # Definição da arquitetura
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        # x vem como [batch_size, window_size]
        print("Input no forward:", x.shape)
        x = x.unsqueeze(-1)  # vira [batch_size, window_size, 1]
        out, _ = self.lstm(x)
        print("Saída do LSTM:", out.shape)
        out = self.fc(out[:, -1, :])
        print("Saída final (fc):", out.shape)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch[:2]  # se vier (X, y, base)
        print("Shapes no training_step -> X:", X.shape, "y:", y.shape)
        outputs = self(X)
        print("Outputs:", outputs.shape)
        loss = torch.nn.functional.l1_loss(outputs.squeeze(-1), y)
        print("Loss shape:", loss.shape)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[:2]  # se o dataset retorna (X, y, base)
        outputs = self(X)  # já está [batch, seq_len, 1]
        outputs = outputs.squeeze(-1)  # vira [batch]
        loss = torch.nn.functional.l1_loss(outputs, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.l1_loss(outputs.squeeze(-1), targets)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        # Adicionado weight_decay para regularização (evita pesos explosivos)
        # Reduzido learning rate para 0.001 para ajuste mais fino
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)


if __name__ == '__main__':

    # Otimização para GPUs NVIDIA (RTX)
    torch.set_float32_matmul_precision('medium')

    # Configurar o loader para encontrar o arquivo
    #base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
    base_path = "."
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw", 
                             processed_path=f"{base_path}/data/processed")
    
    # Obter o caminho do arquivo mais recente dinamicamente
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    print(f"Utilizando arquivo: {filename_path}")

    # Instanciar DataModule (já com scaler interno)
    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",   # coluna com valores de fechamento
        window_size=60,      # Aumentado para 60 dias (aprox 3 meses) para capturar tendências
        batch_size=32        # Aumentado para 32 para estabilizar o cálculo do gradiente
    )

    # Preparar dados
    data_module.setup()

    print(type(data_module))

    # Initialize model
    # Modelo mais robusto com 2 camadas e Dropout
    model = LSTMModel2(hidden_dim=64, num_layers=2, dropout_prob=0.2)

    # --- Callbacks ---
    # Salva o melhor modelo com base na perda de validação
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='lstm-best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # Para o treinamento se a perda de validação não melhorar
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=10, # Número de épocas sem melhora antes de parar
       verbose=True,
       mode='min'
    )

    # Treinamento
    trainer = Trainer(max_epochs=200, accelerator="auto", devices=1,
                        callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, datamodule=data_module)

    # Teste
    print("\nIniciando Teste com o melhor modelo...")
    trainer.test(model, datamodule=data_module)

    # --- PREDIÇÃO E VISUALIZAÇÃO ---
    # O modelo já está com os pesos do melhor checkpoint após o `trainer.fit`
    print("\nGerando gráfico de predições com o melhor modelo...")
    
    model.eval()
    predictions = []
    actuals = []
    bases = []

    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            x, y, base = batch
            out = model(x)

            predictions.extend(out.flatten().tolist())
            actuals.extend(y.flatten().tolist())
            bases.extend(base.flatten().tolist())

    # 🔎 Reconstruir valores reais: Preço = (Norm + 1) * Base
    # Isso reverte a fórmula: norm = (preco / base) - 1
    preds_real = (np.array(predictions) + 1) * np.array(bases)
    actuals_real = (np.array(actuals) + 1) * np.array(bases)

    print(f"preds_real: {preds_real}")
    print(f"actuals_real: {actuals_real}")

    # Plotar
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_real, label='Real (Target)', color='blue')
    plt.plot(preds_real, label='Predição (LSTM)', color='red', linestyle='--')
    plt.title('Predição de Preço de Ações (Conjunto de Teste)')
    plt.legend()
    plt.show()


    if __name__ == "__main__":
    ray.init(num_cpus=15)
    #ray.init(address="auto")   # conecta ao cluster existente
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
    trainer = TorchTrainer(train_func, scaling_config=scaling_config)
    result = trainer.fit()
    print("Fim")
    ray.shutdown()