import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        lag_cols,
        extra_cols,
        target_cols,
        text_col="TaggedNews",
        max_len=64,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.lag_cols = lag_cols
        self.extra_cols = extra_cols
        self.target_cols = target_cols
        self.text_col = text_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _build_lag_tensor(self, row):
        # Convert flat lag columns -> [lag_days, 3]
        # We assume naming: price1_lag1, price2_lag1, price3_lag1, ..., price1_lag7...
        lags = []
        lag_days = len([c for c in self.lag_cols if c.endswith("_lag1")])  # 3 if 3 prices, not lag days
        # safer: detect max lag from names
        max_lag = max(int(c.split("_lag")[-1]) for c in self.lag_cols)

        for lag in range(1, max_lag + 1):
            lags.append([float(row[f"{col}_lag{lag}"]) for col in self.target_cols])

        return torch.tensor(lags, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        lag_tensor = self._build_lag_tensor(row)
        extra_tensor = torch.tensor(row[self.extra_cols].values.astype("float32"), dtype=torch.float32)

        text = str(row[self.text_col])
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        target = torch.tensor(row[self.target_cols].values.astype("float32"), dtype=torch.float32)

        return (
            lag_tensor,
            extra_tensor,
            tokens["input_ids"].squeeze(0),
            tokens["attention_mask"].squeeze(0),
            target,
        )