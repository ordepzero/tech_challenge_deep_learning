# src/services/data_loader.py

import yfinance as yf
import pandas as pd
import os
from typing import Optional
from pathlib import Path
from datetime import datetime

class StockDataLoader:
    """
    Classe para baixar e salvar dados históricos de ativos financeiros
    usando a biblioteca yfinance.
    """

    def __init__(self, 
                 raw_path: str = "data/raw", 
                 processed_path: str = "data/processed"):
        """
        Inicializa o DataLoader.

        Args:
            raw_path (str): Caminho base para salvar dados brutos.
            processed_path (str): Caminho base para salvar dados processados.
        """
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        print(f"Diretório de dados brutos: '{self.raw_path.resolve()}'")
        print(f"Diretório de dados processados: '{self.processed_path.resolve()}'")

    def _get_save_path(self, base_path: Path, ticker: str) -> Path:
        """
        Gera um caminho de arquivo versionado pela data atual (YYYY-MM-DD).

        Se chamado várias vezes no mesmo dia para o mesmo ticker, retornará o mesmo caminho.
        """
        safe_ticker_name = ticker.replace(".", "_")
        today_str = datetime.now().strftime("%Y-%m-%d")
        ticker_dir = base_path / safe_ticker_name
        ticker_dir.mkdir(exist_ok=True)
        return ticker_dir / f"{today_str}.csv"

    def _get_latest_file_path(self, base_path: Path, ticker: str) -> Path:
        """Encontra o arquivo mais recente para um determinado ticker."""
        safe_ticker_name = ticker.replace(".", "_")
        ticker_dir = base_path / safe_ticker_name

        if not ticker_dir.is_dir():
            raise FileNotFoundError(f"Nenhum diretório encontrado para o ticker '{ticker}' em '{base_path}'.")

        files = list(ticker_dir.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado para o ticker '{ticker}' em '{ticker_dir}'.")

        # Ordena os arquivos pela data no nome (mais recente primeiro) e pega o primeiro
        # O nome do arquivo já é a data, então a ordenação de string funciona
        latest_file = sorted(files, key=lambda f: f.name, reverse=True)[0]
        return latest_file

    def save_raw(self, df: pd.DataFrame, ticker: str) -> str:
        """Salva um DataFrame de dados brutos."""
        file_path = self._get_save_path(self.raw_path, ticker)
        df.to_csv(file_path)
        print(f"Dados brutos salvos em: '{file_path}'")
        return file_path
    
    def save_processed(self, df: pd.DataFrame, ticker: str) -> str:
        """Salva um DataFrame de dados processados."""
        file_path = self._get_save_path(self.processed_path, ticker)
        df.to_csv(file_path)
        print(f"Dados processados salvos em: '{file_path}'")
        return file_path

    def load_raw(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Carrega o conjunto de dados brutos mais recente para um ticker."""
        file_path = self._get_latest_file_path(self.raw_path, ticker)
        print(f"Carregando dados brutos de: '{file_path}'")
        df = pd.read_csv(str(file_path), index_col="Date", parse_dates=True)

        if start or end:
            # Garante que o índice seja do tipo Datetime para a filtragem funcionar
            df.index = pd.to_datetime(df.index)
            df = df.loc[start:end]
            print(f"Dados filtrados entre {start or 'início'} e {end or 'fim'}.")

        return df

    def load_processed(self, ticker: str) -> pd.DataFrame:
        """Carrega o conjunto de dados processados mais recente para um ticker."""
        file_path = self._get_latest_file_path(self.processed_path, ticker)
        print(f"Carregando dados processados de: '{file_path}'")
        # Adapte os parâmetros de leitura se o formato processado for diferente
        df = pd.read_csv(str(file_path), index_col=0, parse_dates=True)
        return df

    def get_latest_file_path(self, ticker: str, kind: str = "raw") -> str:
        """
        Retorna o caminho do arquivo mais recente para o ticker especificado.

        Args:
            ticker (str): O símbolo do ativo (ex: "NVDA").
            kind (str): "raw" para dados brutos ou "processed" para processados.

        Returns:
            str: O caminho absoluto do arquivo.
        """
        if kind == "raw":
            base_path = self.raw_path
        elif kind == "processed":
            base_path = self.processed_path
        else:
            raise ValueError("O parâmetro 'kind' deve ser 'raw' ou 'processed'.")

        file_path = self._get_latest_file_path(base_path, ticker)
        return str(file_path.resolve())

    def download_history(
        self,
        ticker: str,
        period: str = "max",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Baixa os dados históricos de um ticker e retorna como um DataFrame.

        Args:
            (mesmos argumentos do método original)

        Returns:
            pd.DataFrame: DataFrame com os dados históricos.
        """
        print(f"Iniciando download para o ticker: '{ticker}'...")
        stock = yf.Ticker(ticker)
        data = stock.history(
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=True,
        )
        if data.empty:
            raise ValueError(f"Nenhum dado encontrado para o ticker '{ticker}' com os parâmetros fornecidos.")
        
        print(f"Download concluído! {len(data)} registros baixados.")
        return data

    def download_and_save_history(
        self,
        ticker: str,
        period: str = "max",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> str:
        """
        Orquestra o download e o salvamento dos dados históricos brutos.

        Args: (mesmos argumentos do método original)
            ticker (str): O símbolo do ativo a ser baixado (ex: "NVDA", "PETR4.SA").
            period (str, optional): O período de dados a ser baixado.
                Valores válidos: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
                Padrão é "max".
            interval (str, optional): O intervalo dos dados.
                Valores válidos: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo".
                Padrão é "1d".
            start (Optional[str], optional): A data de início no formato "YYYY-MM-DD".
                Se fornecido, sobrescreve 'period'. Padrão é None.
            end (Optional[str], optional): A data de fim no formato "YYYY-MM-DD".
                Padrão é None.

        Returns:
            str: O caminho completo para o arquivo CSV salvo.
        """
        try:
            # 1. Baixa os dados
            data = self.download_history(ticker, period, interval, start, end)
            # 2. Salva os dados brutos e retorna o caminho como string
            file_path = self.save_raw(data, ticker)
            return str(file_path)
        except ValueError as ve:
            # Relança o erro de valor para que o chamador possa tratá-lo
            print(f"Erro de validação: {ve}")
            raise
        except Exception as e:
            # Captura outras exceções (ex: rede, erros internos do yfinance)
            error_message = f"Ocorreu um erro inesperado ao baixar dados para '{ticker}': {e}"
            print(error_message)
            raise Exception(error_message) from e



if __name__ == "__main__":

    custom_base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
    
    loader = StockDataLoader(raw_path=f"{custom_base_path}/data/raw", processed_path=f"{custom_base_path}/data/processed")

    # 1: Baixar e salvar dados históricos da API YFinance
    try:
        loader.download_and_save_history(ticker="NVDA", period="1y")
    except Exception as e:
        print(f"Não foi possível baixar os dados da NVDA: {e}")

    print("\n" + "="*50 + "\n")

    # 2: Carregar os dados brutos que acabamos de salvar
    try:
        df_nvda = loader.load_raw(ticker="NVDA")
        print("\nÚltimos 5 registros da NVIDIA:")
        print(df_nvda.tail())
    except Exception as e:
        print(f"Não foi possível carregar os dados da NVDA: {e}")

    print("\n" + "="*50 + "\n")

    # 3: Salvar um DataFrame processado
    try:
        df_nvda_raw = loader.load_raw("NVDA")

        # Simula um processamento: cria uma coluna de média móvel
        df_processed = df_nvda_raw.copy()
        df_processed['SMA_20'] = df_processed['Close'].rolling(window=20).mean()
        loader.save_processed(df_processed, "NVDA")

        # Carrega o dado processado para verificar
        df_check = loader.load_processed("NVDA")
        print("\nÚltimos 5 registros do DataFrame PROCESSADO da NVIDIA:")
        print(df_check.tail())
    except ValueError as e:
        print(f"Erro capturado como esperado: {e}")
