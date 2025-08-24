from fastapi import APIRouter, Query
from models.predictor import StockPredictor
from utils.helpers import get_investment_advice

router = APIRouter()

@router.get("/predict/{symbol}")
def predict_stock(
    symbol: str,
    days: int = Query(1, ge=1, le=30),
    provide_advice: str = Query("false")
):
    # Convert provide_advice to boolean
    provide_advice = str(provide_advice).strip().lower() == "true"

    # Initialize predictor and get predictions
    predictor = StockPredictor(symbol)
    result = predictor.predict(days_ahead=days)

    # Round values
    if "predicted_prices" in result and "current_price" in result:
        try:
            result["predicted_prices"] = [round(float(p), 2) for p in result["predicted_prices"]]
            result["current_price"] = round(float(result["current_price"]), 2)
        except (ValueError, TypeError):
            result["predicted_prices"] = result.get("predicted_prices", [])
            result["current_price"] = result.get("current_price", None)

        # Add advice if requested
        if provide_advice:
            result["investment_advice"] = get_investment_advice(result["predicted_prices"], symbol)

    return result
