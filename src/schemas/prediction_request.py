
from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    data: List[float]
    model_run_id: str
