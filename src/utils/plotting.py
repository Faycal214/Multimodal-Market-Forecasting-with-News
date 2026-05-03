from pathlib import Path

import matplotlib.pyplot as plt

from src.config.config import FIGURES_DIR


def save_training_curves(history, filename="training_curves.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_r2_mean"], label="Val R²")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R²")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path