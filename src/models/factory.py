# src/models/factory.py
from src.models.mlp_model import MLPModel
from src.models.lstm_model import LSTMModel

class ModelFactory:
    """
    Fábrica (Factory) para criação de instâncias de modelos baseada na configuração fornecida.
    """
    @staticmethod
    def create(config):
        """
        Cria e retorna uma instância do modelo especificado na configuração.
        
        Args:
            config: Objeto de configuração (ex: TrainRequest) contendo o 'model_type' e hiperparâmetros.
        
        Returns:
            LightningModule: Uma instância de MLPModel ou LSTMModel.
        """
        params = config.model_dump(exclude={"model_type", "log_level"})
        if config.model_type == "mlp":
            return MLPModel(**params)
        elif config.model_type == "lstm":
            return LSTMModel(**params)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {config.model_type}")
