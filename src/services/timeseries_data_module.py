import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
#import ray
#import numpy as np
#import pandas as pd
from src.services.data_loader import StockDataLoader


# Dataset para séries temporais
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size: int):
        """
        Args:
            series (array-like): valores da série temporal (ex: fechamento de ações).
            window_size (int): número de passos anteriores usados como entrada.
        """
        self.series = series
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        # X = janela de n valores anteriores
        x = self.series[idx:idx+self.window_size]
        # y = próximo valor após a janela
        y = self.series[idx+self.window_size]
        
        # --- Normalização por Janela (Window Normalization) ---
        # ALTERAÇÃO CRÍTICA: Usar o ÚLTIMO valor da janela (x[-1]) como base.
        # Isso transforma o problema em prever o retorno imediato (próximo dia)
        # em vez do retorno acumulado desde o início da janela.
        # Isso corrige o viés de alta (overshooting) herdado de tendências passadas.
        base_value = x[-1] if x[-1] != 0 else 1e-8
        
        x_norm = (x / base_value) - 1
        y_norm = (y / base_value) - 1
        
        # Retornamos também o base_value para conseguir reconstruir o preço real no gráfico
        return torch.tensor(x_norm, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32), torch.tensor(base_value, dtype=torch.float32)


class TimeSeriesDataModule(LightningDataModule):
    def __init__(self, csv_path, value_col, window_size=10, batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.value_col = value_col
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        series = df[self.value_col].values.reshape(-1, 1)

        # Divisão cronológica dos dados
        train_size = int(0.7 * len(series))
        val_size = int(0.15 * len(series))

        train_data = series[:train_size]
        val_data = series[train_size : train_size + val_size]
        test_data = series[train_size + val_size :]

        # Criar os Datasets passando os dados BRUTOS (a normalização ocorre no __getitem__)
        self.train_dataset = TimeSeriesDataset(train_data.flatten(), self.window_size)
        self.val_dataset = TimeSeriesDataset(val_data.flatten(), self.window_size)
        self.test_dataset = TimeSeriesDataset(test_data.flatten(), self.window_size)

    def train_dataloader(self):
        # shuffle=True é geralmente recomendado para que os lotes sejam vistos em ordem aleatória,
        # melhorando a generalização do modelo. Ele embaralha as 'janelas', não os dados dentro delas.
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=2)
    
    def prepare_data(self):
        pass



# class RayTimeSeriesDataModule:
#     """
#     Classe wrapper para gerar Datasets do Ray (ray.data) reutilizando a lógica
#     de pré-processamento do TimeSeriesDataModule, mas retornando numpy arrays
#     em vez de tensores PyTorch (compatível com Arrow).
#     """
#     def __init__(self, csv_path, value_col, window_size=10, batch_size=1):
#         self.csv_path = csv_path
#         self.value_col = value_col
#         self.window_size = window_size
#         self.batch_size = batch_size

#     def _make_dataset(self, series):
#         items = []
#         for idx in range(len(series) - self.window_size):
#             x = series[idx:idx+self.window_size]
#             y = series[idx+self.window_size]

#             base_value = x[-1] if x[-1] != 0 else 1e-8
#             x_norm = (x / base_value) - 1
#             y_norm = (y / base_value) - 1

#             items.append({
#                 "x": x_norm.astype("float32"),
#                 "y": np.float32(y_norm),
#                 "base": np.float32(base_value)
#             })
#         return ray.data.from_items(items)

#     def get_ray_datasets(self):
#         df = pd.read_csv(self.csv_path)
#         series = df[self.value_col].values.reshape(-1)

#         train_size = int(0.7 * len(series))
#         val_size = int(0.15 * len(series))

#         train_data = series[:train_size]
#         val_data = series[train_size : train_size + val_size]
#         test_data = series[train_size + val_size :]

#         return (
#             self._make_dataset(train_data),
#             self._make_dataset(val_data),
#             self._make_dataset(test_data)
#        )

if __name__ == "__main__":
    # Configurar o loader para encontrar o arquivo
    base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw", 
                             processed_path=f"{base_path}/data/processed")
    
    # Obter o caminho do arquivo mais recente dinamicamente
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    print(f"Utilizando arquivo: {filename_path}")

    # Instanciar DataModule
    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",   # coluna com valores de fechamento
        window_size=5,       # usar 5 dias anteriores como entrada
        batch_size=64
    )

    # Preparar dados
    data_module.setup()

    # Obter um batch
    for batch in data_module.train_dataloader():
        X, y = batch
        print(X.shape, y.shape)
        break