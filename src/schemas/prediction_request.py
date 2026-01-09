from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    """
    Representa uma requisição para realizar predições de preços.
    """
    data: List[float] = Field(..., description="Lista de valores de fechamento (série temporal)")
    model_run_id: str = Field(..., description="ID da execução (run) do modelo no MLflow", example="504decf66dd846d498ae1dca9ea51f6d")
    ticker: Optional[str] = Field(None, description="Ticker da ação para completar a janela utilizando dados históricos", example="NVDA")
