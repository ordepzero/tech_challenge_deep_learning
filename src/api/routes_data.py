from src.services.data_loader import StockDataLoader
from fastapi import APIRouter, HTTPException
from enum import Enum
from src.schemas.response import APIResponse

router = APIRouter(prefix="/data", tags=["data"])

class PeriodEnum(str, Enum):
    """
    Enumeração de períodos válidos para o download de dados históricos.
    """
    d1 = "1d"
    d5 = "5d"
    mo1 = "1mo"
    mo3 = "3mo"
    mo6 = "6mo"
    y1 = "1y"
    y2 = "2y"
    y5 = "5y"
    y10 = "10y"
    ytd = "ytd"
    max = "max"

@router.get("/download_history/{ticker}", response_model=APIResponse)
def download_and_save_history(ticker: str, period: PeriodEnum = PeriodEnum.max):
    """
    Realiza o download e o salvamento do histórico de preços de uma ação.
    
    Args:
        ticker (str): O código da ação (ex: NVDA, AAPL).
        period (PeriodEnum): O período de tempo desejado para o histórico.
        
    Returns:
        APIResponse: Resposta indicando sucesso ou erro no download.
    """
    loader = StockDataLoader(raw_path="./data/raw", processed_path="./data/processed")

    try:
        loader.download_and_save_history(ticker=ticker, period=period.value)
        return APIResponse(
            status="success",
            message="Histórico baixado com sucesso",
            data={"ticker": ticker, "period": period.value}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                status="error",
                message="Não foi possível baixar os dados",
                errors={"ticker": ticker, "error": str(e)}
            ).dict()
        )
