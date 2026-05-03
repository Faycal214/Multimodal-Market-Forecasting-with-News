from src.models.train import train_end_to_end
from src.utils.plotting import save_training_curves


def main():
    results = train_end_to_end()

    history = results["history"]

    # Save training curves
    fig_path = save_training_curves(history)

    print("\n========== TRAINING SUMMARY ==========")
    print("Best validation R²:", max(history["val_r2_mean"]))
    print("Epochs run:", len(history["train_loss"]))
    print("Figure saved at:", fig_path)
    print("Processed data saved in: data/processed/")
    print("Models saved in: models/")
    print("=====================================\n")


if __name__ == "__main__":
    main()