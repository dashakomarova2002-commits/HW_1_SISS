import uuid

from fastapi import FastAPI, HTTPException, Query, status

from schemas import FarmRequest, PredictionResponse
from services import build_prediction, is_valid_region
from storage import save_prediction, get_prediction, list_predictions

app = FastAPI(
    title="Agro Scoring API",
    description="API for agricultural risk scoring",
    version="1.0.0"
)


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": "agro-risk-model",
        "model_version": "1.0",
        "model_type": "risk-scoring",
        "status": "ready"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Farm risk assessment"
)
def predict(request: FarmRequest):
    if not is_valid_region(request.region):
        raise HTTPException(status_code=400, detail="Unknown region")

    request_id = str(uuid.uuid4())
    result = build_prediction(request, request_id)
    save_prediction(request_id, result)
    return result


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse
)
def get_prediction_endpoint(request_id: str):
    result = get_prediction(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


@app.get("/predictions")
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: str | None = None
):
    return list_predictions(limit=limit, risk_level=risk_level)