from fastapi import APIRouter
from src.services.data_loader import StockDataLoader

router = APIRouter(
    prefix="/utils",
    tags=["utils"]
)
