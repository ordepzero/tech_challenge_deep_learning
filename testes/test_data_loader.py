import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Adiciona o diretório 'src' ao path para que possamos importar o FinancialDataLoader
import sys

# Navega para o diretório raiz do projeto para que a importação de 'src' funcione
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.services.data_loader import FinancialDataLoader

class TestFinancialDataLoader(unittest.TestCase):

    def setUp(self):
        """
        Configura um ambiente de teste limpo antes de cada teste.
        Cria um diretório temporário para os dados.
        """
        self.test_dir = tempfile.mkdtemp()
        self.raw_path = Path(self.test_dir) / "raw"
        self.processed_path = Path(self.test_dir) / "processed"
        self.loader = FinancialDataLoader(raw_path=str(self.raw_path), processed_path=str(self.processed_path))

    def tearDown(self):
        """
        Limpa o ambiente de teste após cada teste.
        Remove o diretório temporário.
        """
        shutil.rmtree(self.test_dir)

    def test_initialization_creates_directories(self):
        """Testa se os diretórios raw e processed são criados na inicialização."""
        self.assertTrue(self.raw_path.exists())
        self.assertTrue(self.raw_path.is_dir())
        self.assertTrue(self.processed_path.exists())
        self.assertTrue(self.processed_path.is_dir())

    @patch('yfinance.Ticker')
    def test_download_history_success(self, mock_ticker):
        """Testa o download bem-sucedido de dados históricos (com mock)."""
        # Configura o mock para retornar um DataFrame de exemplo
        mock_data = pd.DataFrame({'Close': [100, 101, 102]})
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_instance

        # Executa o método
        df = self.loader.download_history(ticker="MOCK.SA")

        # Verifica se o método history foi chamado
        mock_instance.history.assert_called_once()
        # Verifica se o DataFrame retornado é o esperado
        pd.testing.assert_frame_equal(df, mock_data)

    @patch('yfinance.Ticker')
    def test_download_history_no_data(self, mock_ticker):
        """Testa o comportamento quando yfinance retorna um DataFrame vazio."""
        # Configura o mock para retornar um DataFrame vazio
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        # Verifica se um ValueError é levantado
        with self.assertRaises(ValueError):
            self.loader.download_history(ticker="FAKE.SA")

    def test_save_and_load_raw(self):
        """Testa o salvamento e carregamento de dados brutos."""
        ticker = "TEST.SA"
        data = pd.DataFrame({'Price': [10, 11]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        data.index.name = "Date"

        # Salva os dados
        saved_path_str = self.loader.save_raw(data, ticker)
        saved_path = Path(saved_path_str)

        # Verifica se o arquivo foi criado no local esperado
        self.assertTrue(saved_path.exists())
        today_str = datetime.now().strftime("%Y-%m-%d")
        expected_path = self.raw_path / "TEST_SA" / f"{today_str}.csv"
        self.assertEqual(saved_path, expected_path)

        # Carrega os dados e verifica se são iguais
        loaded_data = self.loader.load_raw(ticker)
        pd.testing.assert_frame_equal(data, loaded_data)

    def test_save_and_load_processed(self):
        """Testa o salvamento e carregamento de dados processados."""
        ticker = "PROC.SA"
        data = pd.DataFrame({'SMA_20': [9.5, 10.5]})

        # Salva e carrega
        self.loader.save_processed(data, ticker)
        loaded_data = self.loader.load_processed(ticker)

        pd.testing.assert_frame_equal(data, loaded_data)

    def test_get_latest_file_path(self):
        """Testa se o método encontra o arquivo mais recente corretamente."""
        ticker = "LATEST.SA"
        ticker_dir = self.raw_path / "LATEST_SA"
        ticker_dir.mkdir()

        # Cria arquivos com datas passadas
        (ticker_dir / "2023-10-26.csv").touch()
        (ticker_dir / "2023-10-27.csv").touch() # O mais recente
        (ticker_dir / "2023-10-25.csv").touch()

        latest_path = self.loader._get_latest_file_path(self.raw_path, ticker)
        self.assertEqual(latest_path.name, "2023-10-27.csv")

    def test_load_raw_with_date_filter(self):
        """Testa a filtragem por data ao carregar dados brutos."""
        ticker = "FILTER.SA"
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
        data = pd.DataFrame({'Price': [10, 11, 12, 13]}, index=dates)
        data.index.name = "Date"
        self.loader.save_raw(data, ticker)

        # Carrega com filtro de data
        filtered_data = self.loader.load_raw(ticker, start="2023-01-02", end="2023-01-03")

        # Cria o DataFrame esperado após o filtro
        expected_data = data.loc["2023-01-02":"2023-01-03"]

        self.assertEqual(len(filtered_data), 2)
        pd.testing.assert_frame_equal(filtered_data, expected_data)

    def test_load_file_not_found(self):
        """Testa se FileNotFoundError é levantado quando não há dados para carregar."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_raw("NONEXISTENT.SA")

        with self.assertRaises(FileNotFoundError):
            self.loader.load_processed("NONEXISTENT.SA")


if __name__ == '__main__':
    # Para executar os testes, navegue até o diretório raiz do projeto e execute:
    # python -m unittest tests/services/test_data_loader.py
    # ou para descobrir e rodar todos os testes:
    # python -m unittest discover tests
    unittest.main()