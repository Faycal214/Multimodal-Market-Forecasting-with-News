from datetime import date
from typing import List

import json
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.config.config import (
    DEVICE,
    MODEL_BUNDLE_FILE,
    LAG_SCALER_FILE,
    EXTRA_SCALER_FILE,
    TARGET_SCALER_FILE,
    ARTIFACTS_FILE,
)
from src.features.feature_engineering import make_feature_row
from src.models.model import MultimodalForecaster

app = FastAPI(
    title="NewsAware Market Forecasting API",
    description="Predict price1, price2, price3 using time series history + news headlines.",
    version="1.0.0",
)


class HistoryPoint(BaseModel):
    Date: date
    price1: float
    price2: float
    price3: float


class PredictRequest(BaseModel):
    current_date: date = Field(..., description="Date of the prediction target")
    news: str = Field("", description="News headlines aggregated for the current date")
    history: List[HistoryPoint] = Field(..., description="Past rows used to build lag features")


class PredictResponse(BaseModel):
    price1: float
    price2: float
    price3: float


_artifacts = None
_lag_scaler = None
_extra_scaler = None
_target_scaler = None
_checkpoint = None
_model = None
_tokenizer = None


def _load_once():
    global _artifacts, _lag_scaler, _extra_scaler, _target_scaler, _checkpoint, _model, _tokenizer

    if _model is None:
        with open(ARTIFACTS_FILE, "r", encoding="utf-8") as f:
            _artifacts = json.load(f)

        _lag_scaler = joblib.load(LAG_SCALER_FILE)
        _extra_scaler = joblib.load(EXTRA_SCALER_FILE)
        _target_scaler = joblib.load(TARGET_SCALER_FILE)
        _checkpoint = torch.load(MODEL_BUNDLE_FILE, map_location=DEVICE)

        _model = MultimodalForecaster(
            text_model_name=_artifacts["text_model_name"],
            price_dim=len(_artifacts["target_cols"]),
            extra_dim=len(_artifacts["extra_cols"]),
            freeze_text_encoder=False,
        ).to(DEVICE)

        _model.load_state_dict(_checkpoint["model_state_dict"])
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(_artifacts["text_model_name"])

    return _artifacts, _lag_scaler, _extra_scaler, _target_scaler, _model, _tokenizer


@app.get("/")
def root():
    return {"message": "NewsAware Market Forecasting API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(payload: PredictRequest):
    artifacts, lag_scaler, extra_scaler, target_scaler, model, tokenizer = _load_once()

    if len(payload.history) < artifacts["lag_days"]:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough history. Need at least {artifacts['lag_days']} rows."
        )

    history_df = pd.DataFrame([h.model_dump() for h in payload.history])
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    history_df = history_df.sort_values("Date").reset_index(drop=True)

    feat = make_feature_row(
        history_prices=history_df[["Date", "price1", "price2", "price3"]],
        current_date=payload.current_date,
        current_text=payload.news,
        price_cols=artifacts["target_cols"],
        lag_days=artifacts["lag_days"],
    )

    if feat is None:
        raise HTTPException(status_code=400, detail="Could not build features from the provided history.")

    sample_df = pd.DataFrame([feat])

    sample_df[artifacts["lag_cols"]] = lag_scaler.transform(sample_df[artifacts["lag_cols"]])
    sample_df[artifacts["extra_cols"]] = extra_scaler.transform(sample_df[artifacts["extra_cols"]])

    lag_days = artifacts["lag_days"]
    target_cols = artifacts["target_cols"]

    lag_tensor = []
    for lag in range(1, lag_days + 1):
        lag_tensor.append([sample_df.iloc[0][f"{col}_lag{lag}"] for col in target_cols])
    lag_tensor = torch.tensor([lag_tensor], dtype=torch.float32, device=DEVICE)

    extra_values = sample_df[artifacts["extra_cols"]].iloc[0].to_numpy(dtype="float32")
    extra_tensor = torch.tensor(np.array([extra_values]), dtype=torch.float32, device=DEVICE)

    tokens = tokenizer(
        str(payload.news),
        padding="max_length",
        truncation=True,
        max_length=artifacts["max_len"],
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    pred_scaled = model(lag_tensor, extra_tensor, input_ids, attention_mask).cpu().numpy()
    pred = target_scaler.inverse_transform(pred_scaled)[0]

    return PredictResponse(
        price1=float(pred[0]),
        price2=float(pred[1]),
        price3=float(pred[2]),
    )