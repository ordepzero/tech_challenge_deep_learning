from enum import Enum
from pydantic import BaseModel

class ModelType(str, Enum):
    lstm = "lstm"
    mlp = "mlp"

class TrainRequest(BaseModel):
    model_type: ModelType
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_prob: float = 0.2
    learning_rate: float = 0.001
    window_size: int = 60