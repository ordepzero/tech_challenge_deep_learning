import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from src.services.data_loader import StockDataLoader


# Dataset para séries temporais
class TimeSeriesDataset(Dataset):
    """
    Dataset customizado para séries temporais que aplica normalização baseada na janela.
    Estende a classe Dataset do PyTorch.
    """
    def __init__(self, series, window_size: int):
        """
        Inicializa o dataset.
        
        Args:
            series (array-like): Valores da série temporal (ex: preços de fechamento).
            window_size (int): Tamanho da janela de passos de tempo anteriores usados como entrada.
        """
        self.series = series
        self.window_size = window_size

    def __len__(self):
        """
        Retorna o número de amostras disponíveis (total de elementos menos o tamanho da janela).
        """
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        """
        Gera uma amostra (janela de entrada, alvo e valor base).
        
        Aplica "Window Normalization" utilizando o último valor da janela de entrada como base,
        o que permite ao modelo prever retornos relativos.
        """
        # X = janela de n valores anteriores
        x = self.series[idx:idx+self.window_size]
        # y = próximo valor imediatamente após a janela
        y = self.series[idx+self.window_size]
        
        # --- Normalização por Janela (Window Normalization) ---
        # Utiliza o ÚLTIMO valor da janela (x[-1]) como base.
        base_value = x[-1] if x[-1] != 0 else 1e-8
        
        x_norm = (x / base_value) - 1
        y_norm = (y / base_value) - 1
        
        # Retorna X normalizado, Y alvo normalizado e o base_value para reconstrução posterior do preço real
        return torch.tensor(x_norm, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32), torch.tensor(base_value, dtype=torch.float32)


class TimeSeriesDataModule(LightningDataModule):
    """
    DataModule do PyTorch Lightning para gerenciar o carregamento e divisão dos dados de séries temporais.
    """
    def __init__(self, csv_path, value_col, window_size=10, batch_size=32):
        """
        Args:
            csv_path (str): Caminho do arquivo CSV contendo os dados.
            value_col (str): Nome da coluna a ser utilizada como série temporal.
            window_size (int): Tamanho da janela deslizante para entrada do modelo.
            batch_size (int): Tamanho do lote para os DataLoaders.
        """
        super().__init__()
        self.csv_path = csv_path
        self.value_col = value_col
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Lê o arquivo CSV e divide os dados cronologicamente em Treino (70%), Validação (15%) e Teste (15%).
        """
        df = pd.read_csv(self.csv_path)
        series = df[self.value_col].values.reshape(-1, 1)

        # Divisão cronológica dos dados
        train_size = int(0.7 * len(series))
        val_size = int(0.15 * len(series))

        train_data = series[:train_size]
        val_data = series[train_size : train_size + val_size]
        test_data = series[train_size + val_size :]

        # Cria as instâncias do TimeSeriesDataset (a normalização ocorre sob demanda no __getitem__)
        self.train_dataset = TimeSeriesDataset(train_data.flatten(), self.window_size)
        self.val_dataset = TimeSeriesDataset(val_data.flatten(), self.window_size)
        self.test_dataset = TimeSeriesDataset(test_data.flatten(), self.window_size)

    def train_dataloader(self):
        """
        Retorna o DataLoader para o conjunto de treinamento com embaralhamento habilitado.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        """
        Retorna o DataLoader para o conjunto de validação.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        """
        Retorna o DataLoader para o conjunto de teste.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
    
    def prepare_data(self):
        """
        Método reservado para operações de preparação que não dependem do estado da GPU (ex: downloads).
        """
        pass



if __name__ == "__main__":
    # Script de teste para validação local do carregamento de dados
    base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
    loader = StockDataLoader(raw_path=f"{base_path}/data/raw", 
                             processed_path=f"{base_path}/data/processed")
    
    # Busca o arquivo mais recente da NVDA para testes
    filename_path = loader.get_latest_file_path("NVDA", kind="raw")
    print(f"Utilizando arquivo: {filename_path}")

    # Inicializa o DataModule
    data_module = TimeSeriesDataModule(
        csv_path=filename_path,
        value_col="Close",
        window_size=5,
        batch_size=64
    )

    data_module.setup()

    # Exibe o formato do primeiro lote de dados
    for batch in data_module.train_dataloader():
        X, y, base = batch
        print(f"Formato de X (Entrada): {X.shape}, Formato de y (Alvo): {y.shape}")
        break