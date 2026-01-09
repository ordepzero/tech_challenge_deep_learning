# src/services/data_loader.py

import yfinance as yf
import pandas as pd
import os
from typing import Optional
from pathlib import Path
from datetime import datetime
import logging

# Configuração do logger para o módulo
logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    Classe para baixar e salvar dados históricos de ativos financeiros
    usando a biblioteca yfinance. Gerencia diretórios brutos (raw) e processados.
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
        logger.info(f"Diretório de dados brutos: '{self.raw_path.resolve()}'")
        logger.info(f"Diretório de dados processados: '{self.processed_path.resolve()}'")

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
        """
        Encontra o arquivo CSV mais recente para um determinado ticker dentro de um diretório base.
        """
        safe_ticker_name = ticker.replace(".", "_")
        ticker_dir = base_path / safe_ticker_name

        if not ticker_dir.is_dir():
            raise FileNotFoundError(f"Nenhum diretório encontrado para o ticker '{ticker}' em '{base_path}'.")

        files = list(ticker_dir.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado para o ticker '{ticker}' em '{ticker_dir}'.")

        # Ordena os arquivos pela data no nome (mais recente primeiro)
        latest_file = sorted(files, key=lambda f: f.name, reverse=True)[0]
        return latest_file

    def save_raw(self, df: pd.DataFrame, ticker: str) -> str:
        """
        Salva um DataFrame de dados brutos em formato CSV.
        """
        file_path = self._get_save_path(self.raw_path, ticker)
        df.to_csv(file_path)
        logger.info(f"Dados brutos salvos em: '{file_path}'")
        return file_path
    
    def save_processed(self, df: pd.DataFrame, ticker: str) -> str:
        """
        Salva um DataFrame de dados processados em formato CSV.
        """
        file_path = self._get_save_path(self.processed_path, ticker)
        df.to_csv(file_path)
        logger.info(f"Dados processados salvos em: '{file_path}'")
        return file_path

    def load_raw(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega o conjunto de dados brutos mais recente para um ticker, com opção de filtro de datas.
        """
        file_path = self._get_latest_file_path(self.raw_path, ticker)
        logger.info(f"Carregando dados brutos de: '{file_path}'")
        df = pd.read_csv(str(file_path), index_col="Date", parse_dates=True)

        if start or end:
            df.index = pd.to_datetime(df.index)
            df = df.loc[start:end]
            logger.info(f"Dados filtrados entre {start or 'início'} e {end or 'fim'}.")

        return df

    def load_processed(self, ticker: str) -> pd.DataFrame:
        """
        Carrega o conjunto de dados processados mais recente para um ticker.
        """
        file_path = self._get_latest_file_path(self.processed_path, ticker)
        logger.info(f"Carregando dados processados de: '{file_path}'")
        df = pd.read_csv(str(file_path), index_col=0, parse_dates=True)
        return df

    def get_latest_file_path(self, ticker: str, kind: str = "raw") -> str:
        """
        Retorna o caminho absoluto do arquivo mais recente para o ticker especificado.
        Args:
            ticker (str): Símbolo do ativo.
            kind (str): Tipo dos dados ('raw' ou 'processed').
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
        Baixa os dados históricos de um ticker via yfinance e retorna como um DataFrame.
        """
        logger.info(f"Iniciando download para o ticker: '{ticker}'...")
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
        
        logger.info(f"Download concluído! {len(data)} registros baixados.")
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
        """
        try:
            # 1. Baixa os dados
            data = self.download_history(ticker, period, interval, start, end)
            # 2. Salva os dados brutos e retorna o caminho como string
            file_path = self.save_raw(data, ticker)
            return str(file_path)
        except ValueError as ve:
            logger.error(f"Erro de validação: {ve}")
            raise
        except Exception as e:
            error_message = f"Ocorreu um erro inesperado ao baixar dados para '{ticker}': {e}"
            logger.error(error_message)
            raise Exception(error_message) from e



if __name__ == "__main__":
    # Exemplo de uso e teste das funções do DataLoader
    logging.basicConfig(level=logging.INFO)
    custom_base_path = r"D:\arquivos_antigos\Projetos\Alura\DeepLearning_pytorch\stock_price_prediction"
    
    loader = StockDataLoader(raw_path=f"{custom_base_path}/data/raw", processed_path=f"{custom_base_path}/data/processed")

    # Baixar e salvar dados históricos
    try:
        loader.download_and_save_history(ticker="NVDA", period="1y")
    except Exception as e:
        logger.error(f"Não foi possível baixar os dados da NVDA: {e}")

    logger.info("="*50)

    # Carregar os dados brutos
    try:
        df_nvda = loader.load_raw(ticker="NVDA")
        logger.info("Últimos 5 registros da NVIDIA:")
        logger.info(df_nvda.tail())
    except Exception as e:
        logger.error(f"Não foi possível carregar os dados da NVDA: {e}")

    logger.info("="*50)

    # Simular processamento e salvar
    try:
        df_nvda_raw = loader.load_raw("NVDA")
        df_processed = df_nvda_raw.copy()
        df_processed['SMA_20'] = df_processed['Close'].rolling(window=20).mean()
        loader.save_processed(df_processed, "NVDA")

        df_check = loader.load_processed("NVDA")
        logger.info("Últimos 5 registros do DataFrame PROCESSADO da NVIDIA:")
        logger.info(df_check.tail())
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")

