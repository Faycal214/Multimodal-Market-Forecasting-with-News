import json

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.config.config import (
    DEVICE,
    MODEL_BUNDLE_FILE,
    LAG_SCALER_FILE,
    EXTRA_SCALER_FILE,
    TARGET_SCALER_FILE,
    ARTIFACTS_FILE,
)
from src.data.ingestion import load_train_raw, load_test_raw
from src.features.feature_engineering import make_feature_row
from src.models.model import MultimodalForecaster


def load_artifacts():
    with open(ARTIFACTS_FILE, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    lag_scaler = joblib.load(LAG_SCALER_FILE)
    extra_scaler = joblib.load(EXTRA_SCALER_FILE)
    target_scaler = joblib.load(TARGET_SCALER_FILE)
    checkpoint = torch.load(MODEL_BUNDLE_FILE, map_location=DEVICE)

    return artifacts, lag_scaler, extra_scaler, target_scaler, checkpoint


def build_model(artifacts):
    model = MultimodalForecaster(
        text_model_name=artifacts["text_model_name"],
        price_dim=len(artifacts["target_cols"]),
        extra_dim=len(artifacts["extra_cols"]),
        freeze_text_encoder=False,
    ).to(DEVICE)
    return model


@torch.no_grad()
def recursive_predict(test_prices_path=None, test_news_path=None):
    artifacts, lag_scaler, extra_scaler, target_scaler, checkpoint = load_artifacts()

    train_df = load_train_raw()
    test_df = load_test_raw(test_prices_path, test_news_path)

    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(artifacts["text_model_name"])
    model = build_model(artifacts)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history = train_df[["Date", "price1", "price2", "price3"]].copy()
    predictions = []

    for _, row in test_df.iterrows():
        current_date = row["Date"]
        current_text = row.get("TaggedNews", "")

        feat = make_feature_row(
            history_prices=history,
            current_date=current_date,
            current_text=current_text,
            price_cols=artifacts["target_cols"],
            lag_days=artifacts["lag_days"],
        )

        if feat is None:
            raise ValueError("Not enough history to build inference features.")

        sample_df = pd.DataFrame([feat])

        sample_df[artifacts["lag_cols"]] = lag_scaler.transform(sample_df[artifacts["lag_cols"]])
        sample_df[artifacts["extra_cols"]] = extra_scaler.transform(sample_df[artifacts["extra_cols"]])

        lag_days = artifacts["lag_days"]
        target_cols = artifacts["target_cols"]

        lag_tensor = []
        for lag in range(1, lag_days + 1):
            lag_tensor.append([sample_df.iloc[0][f"{col}_lag{lag}"] for col in target_cols])
        lag_tensor = torch.tensor([lag_tensor], dtype=torch.float32, device=DEVICE)

        extra_tensor = torch.tensor(
            [sample_df[artifacts["extra_cols"]].iloc[0].values.astype("float32")],
            dtype=torch.float32,
            device=DEVICE,
        )

        tokens = tokenizer(
            str(current_text),
            padding="max_length",
            truncation=True,
            max_length=artifacts["max_len"],
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        pred_scaled = model(lag_tensor, extra_tensor, input_ids, attention_mask).cpu().numpy()
        pred = target_scaler.inverse_transform(pred_scaled)[0]

        predictions.append(
            {
                "Date": current_date,
                "price1": float(pred[0]),
                "price2": float(pred[1]),
                "price3": float(pred[2]),
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [{
                        "Date": current_date,
                        "price1": float(pred[0]),
                        "price2": float(pred[1]),
                        "price3": float(pred[2]),
                    }]
                ),
            ],
            ignore_index=True,
        ).sort_values("Date").reset_index(drop=True)

    return pd.DataFrame(predictions)


def save_submission(output_path="submission.csv", test_prices_path=None, test_news_path=None):
    preds = recursive_predict(test_prices_path=test_prices_path, test_news_path=test_news_path)
    preds.to_csv(output_path, index=False)
    return preds