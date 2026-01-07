
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    data: List[float]
    model_run_id: str = Field(..., description="ID do run/modelo no MLflow", example="504decf66dd846d498ae1dca9ea51f6d")
    ticker: Optional[str] = Field(None, description="Ticker para completar a janela historicamente (ex: NVDA)", example="NVDA")
