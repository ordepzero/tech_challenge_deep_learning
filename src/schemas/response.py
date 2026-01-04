from pydantic import BaseModel
from typing import Any, Optional

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None
    errors: Optional[Any] = None