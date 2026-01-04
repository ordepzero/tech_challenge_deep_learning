# src/models/factory.py
from src.models.mlp_model import MLPModel
from src.models.lstm_model import LSTMModel

class ModelFactory:
    @staticmethod
    def create(config):
        params = config.model_dump(exclude={"model_type", "log_level"})
        if config.model_type == "mlp":
            return MLPModel(**params)
        elif config.model_type == "lstm":
            return LSTMModel(**params)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {config.model_type}")