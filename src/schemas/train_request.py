from enum import Enum
from pydantic import BaseModel, Field

class ModelType(str, Enum):
    """
    Enumeração dos tipos de modelos suportados pela plataforma.
    """
    lstm = "lstm"
    mlp = "mlp"

class TrainRequest(BaseModel):
    """
    Define os parâmetros necessários para iniciar um treinamento ou sintonização de hiperparâmetros.
    """
    model_type: ModelType = Field(..., description="Tipo de arquitetura (lstm ou mlp)")
    hidden_dim: int = Field(64, description="Dimensão da camada oculta")
    num_layers: int = Field(2, description="Número de camadas na rede")
    dropout_prob: float = Field(0.2, description="Probabilidade de dropout para regularização")
    learning_rate: float = Field(0.001, description="Taxa de aprendizado inicial")
    window_size: int = Field(60, description="Tamanho da janela de tempo (número de dias anteriores)")
    log_level: str = Field("INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")
