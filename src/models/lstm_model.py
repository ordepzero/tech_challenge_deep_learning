import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule, Trainer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.services.data_loader import StockDataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.services.timeseries_data_module import TimeSeriesDataModule


class LSTMModel(LightningModule):
    """
    Módulo PyTorch Lightning para o modelo LSTM (Long Short-Term Memory).
    Projetado para previsão de séries temporais financeiras.
    """
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
        """
        Inicializa o modelo LSTM com os hiperparâmetros fornecidos.
        """
        super().__init__()
        # Salva todos os hiperparâmetros automaticamente em self.hparams
        self.save_hyperparameters()

        # Definição da arquitetura: Camada LSTM seguida por uma camada Linear (Fully Connected)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        """
        Executa a passagem para frente (forward pass) do modelo.
        """
        # x chega com o formato [batch_size, window_size]
        # print("Input no forward:", x.shape) # Comentado para reduzir ruído nos logs
        x = x.unsqueeze(-1)  # vira [batch_size, window_size, 1]
        out, _ = self.lstm(x)
        # out: [batch_size, seq_len, hidden_size]
        # Pega apenas a saída do último passo de tempo (last time step)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        """
        Executa um único passo de treinamento.
        """
        X, y = batch[:2]  # batch pode retornar (X, y, base)
        outputs = self(X)
        # Calcula a perda utilizando o Erro Médio Absoluto (MAE/L1 Loss)
        loss = torch.nn.functional.l1_loss(outputs.squeeze(-1), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executa um único passo de validação.
        """
        X, y = batch[:2]
        outputs = self(X) 
        outputs = outputs.squeeze(-1)
        loss = torch.nn.functional.l1_loss(outputs, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Executa um único passo de teste.
        """
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.l1_loss(outputs.squeeze(-1), targets)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Configura o otimizador da rede (Adam com weight decay).
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)


if __name__ == '__main__':
    # Script de treinamento local e visualização de resultados

    # Otimização para GPUs NVIDIA
    torch.set_float32_matmul_precision('medium')

    base_path = "."
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw", 
                             processed_path=f"{base_path}/data/processed")
    
    # Busca o arquivo de dados mais recente
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    print(f"Utilizando arquivo: {filename_path}")

    # Inicializa o DataModule
    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",
        window_size=60,
        batch_size=32
    )
    data_module.setup()

    # Inicializa o modelo
    model = LSTMModel(hidden_dim=64, num_layers=2, dropout_prob=0.2)

    # Configuração de Checkpoint para salvar o melhor modelo
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='lstm-best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    # Configuração de Early Stopping para evitar overfitting
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=10,
       verbose=True,
       mode='min'
    )

    # Inicializa o Trainer do Lightning
    trainer = Trainer(max_epochs=200, accelerator="auto", devices=1,
                        callbacks=[checkpoint_callback, early_stop_callback])
    
    # Inicia o treinamento
    trainer.fit(model, datamodule=data_module)

    # Inicia os testes
    print("\nIniciando Teste com o melhor modelo...")
    trainer.test(model, datamodule=data_module)

    # --- PREDIÇÃO E VISUALIZAÇÃO ---
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

    # Reconstroi os valores reais de preço: Preço = (Retorno_Normalizado + 1) * Valor_Base
    preds_real = (np.array(predictions) + 1) * np.array(bases)
    actuals_real = (np.array(actuals) + 1) * np.array(bases)

    # Plotagem dos resultados
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_real, label='Real (Target)', color='blue')
    plt.plot(preds_real, label='Predição (LSTM)', color='red', linestyle='--')
    plt.title('Predição de Preço de Ações (Conjunto de Teste)')
    plt.xlabel('Tempo')
    plt.ylabel('Preço ($)')
    plt.legend()
    plt.show()
