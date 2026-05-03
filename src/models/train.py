import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config.config import (
    DEVICE,
    TEXT_MODEL_NAME,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    VAL_SIZE,
    RANDOM_SEED,
    GRAD_CLIP_NORM,
    TARGET_COLS,
    MODEL_BUNDLE_FILE,
    LAG_SCALER_FILE,
    EXTRA_SCALER_FILE,
    TARGET_SCALER_FILE,
    ARTIFACTS_FILE,
    LAG_DAYS,
    MAX_LEN,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    PROCESSED_TRAIN_FILE,
    PROCESSED_DIR,
)
from src.data.ingestion import load_train_raw
from src.data.validation import validate_raw_merged_frame
from src.data.dataset import MarketDataset
from src.features.feature_engineering import build_supervised_frame
from src.models.model import MultimodalForecaster
from src.models.evaluate import evaluate_model


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chronological_split(df, val_size=VAL_SIZE):
    df = df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_size))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, val_df


def fit_scalers(train_df, lag_cols, extra_cols, target_cols):
    lag_scaler = StandardScaler()
    extra_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_df = train_df.copy()
    train_df[lag_cols] = lag_scaler.fit_transform(train_df[lag_cols])
    train_df[extra_cols] = extra_scaler.fit_transform(train_df[extra_cols])
    train_df[target_cols] = target_scaler.fit_transform(train_df[target_cols])

    return train_df, lag_scaler, extra_scaler, target_scaler


def apply_scalers(df, lag_cols, extra_cols, target_cols, lag_scaler, extra_scaler, target_scaler):
    df = df.copy()
    df[lag_cols] = lag_scaler.transform(df[lag_cols])
    df[extra_cols] = extra_scaler.transform(df[extra_cols])
    df[target_cols] = target_scaler.transform(df[target_cols])
    return df


def make_loaders(train_df, val_df, tokenizer, lag_cols, extra_cols):
    train_ds = MarketDataset(
        train_df,
        tokenizer=tokenizer,
        lag_cols=lag_cols,
        extra_cols=extra_cols,
        target_cols=TARGET_COLS,
        max_len=MAX_LEN,
    )
    val_ds = MarketDataset(
        val_df,
        tokenizer=tokenizer,
        lag_cols=lag_cols,
        extra_cols=extra_cols,
        target_cols=TARGET_COLS,
        max_len=MAX_LEN,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for lag_seq, extra_feats, input_ids, attention_mask, y in tqdm(loader, desc="Training", leave=False):
        lag_seq = lag_seq.to(device)
        extra_feats = extra_feats.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(lag_seq, extra_feats, input_ids, attention_mask)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def fit_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_r2_mean": [],
        "val_r2_per_target": [],
    }

    best_r2 = -1e9
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)

        scheduler.step(val_metrics["val_loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_r2_mean"].append(val_metrics["val_r2_mean"])
        history["val_r2_per_target"].append(val_metrics["val_r2_per_target"])

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_r2_mean={val_metrics['val_r2_mean']:.4f} | "
            f"val_r2={val_metrics['val_r2_per_target']}"
        )

        improved = val_metrics["val_r2_mean"] > (best_r2 + MIN_DELTA)
        if improved:
            best_r2 = val_metrics["val_r2_mean"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "best_r2": best_r2,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val R²: {best_r2:.4f}")
            break

    return history, best_state


def save_artifacts(model_name, lag_cols, extra_cols):
    MODEL_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "text_model_name": model_name,
        "target_cols": TARGET_COLS,
        "lag_days": LAG_DAYS,
        "max_len": MAX_LEN,
        "lag_cols": lag_cols,
        "extra_cols": extra_cols,
    }

    with open(ARTIFACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    return artifacts


def train_end_to_end(train_prices_path=None, train_news_path=None):
    set_seed()

    merged_df = load_train_raw(train_prices_path, train_news_path)
    validate_raw_merged_frame(merged_df)

    feature_df, lag_cols, extra_cols = build_supervised_frame(merged_df, price_cols=TARGET_COLS, lag_days=LAG_DAYS)

    train_df, val_df = chronological_split(feature_df, val_size=VAL_SIZE)
    train_df, lag_scaler, extra_scaler, target_scaler = fit_scalers(train_df, lag_cols, extra_cols, TARGET_COLS)
    val_df = apply_scalers(val_df, lag_cols, extra_cols, TARGET_COLS, lag_scaler, extra_scaler, target_scaler)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_FILE, index=False)
    val_df.to_csv(PROCESSED_DIR / "val_features.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    train_loader, val_loader = make_loaders(train_df, val_df, tokenizer, lag_cols, extra_cols)

    model = MultimodalForecaster(
        text_model_name=TEXT_MODEL_NAME,
        price_dim=len(TARGET_COLS),
        extra_dim=len(extra_cols),
        freeze_text_encoder=True,
    ).to(DEVICE)

    history, best_state = fit_model(model, train_loader, val_loader, DEVICE, num_epochs=NUM_EPOCHS)
    model.load_state_dict(best_state["model_state_dict"])

    MODEL_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_BUNDLE_FILE)
    joblib.dump(lag_scaler, LAG_SCALER_FILE)
    joblib.dump(extra_scaler, EXTRA_SCALER_FILE)
    joblib.dump(target_scaler, TARGET_SCALER_FILE)

    artifacts = save_artifacts(TEXT_MODEL_NAME, lag_cols, extra_cols)

    return {
        "model": model,
        "history": history,
        "artifacts": artifacts,
        "feature_df": feature_df,
        "train_df": train_df,
        "val_df": val_df,
    }