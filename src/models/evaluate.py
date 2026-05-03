import numpy as np
import torch
from sklearn.metrics import r2_score


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores)), scores


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = torch.nn.MSELoss()

    preds = []
    targets = []
    total_loss = 0.0

    for lag_seq, extra_feats, input_ids, attention_mask, y in dataloader:
        lag_seq = lag_seq.to(device)
        extra_feats = extra_feats.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        out = model(lag_seq, extra_feats, input_ids, attention_mask)
        loss = criterion(out, y)

        total_loss += loss.item()
        preds.append(out.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mean_r2, per_target = compute_r2(targets, preds)

    return {
        "val_loss": total_loss / max(len(dataloader), 1),
        "val_r2_mean": mean_r2,
        "val_r2_per_target": per_target,
        "preds": preds,
        "targets": targets,
    }