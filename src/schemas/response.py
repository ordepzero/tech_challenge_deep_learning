from pydantic import BaseModel
from typing import Any, Optional

class APIResponse(BaseModel):
    """
    Estrutura padrão para todas as respostas da API.
    """
    status: str = "success"
    message: str
    data: Optional[Any] = None
    errors: Optional[Any] = None
