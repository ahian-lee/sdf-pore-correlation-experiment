#!/usr/bin/env python
import argparse
import json
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D CNN regressor on SDF grids.")
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--property_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--targets", nargs="+", default=["PLD", "LCD"])
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matched_dataframe(sdf_dir: str, property_csv: str, targets: List[str], limit: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(property_csv, usecols=["name"] + targets).dropna().drop_duplicates("name")
    matched = []
    for name in df["name"]:
        path = os.path.join(sdf_dir, f"{name}.npy")
        if os.path.exists(path):
            matched.append(name)
    out = df[df["name"].isin(matched)].reset_index(drop=True)
    if limit > 0 and len(out) > limit:
        out = out.sample(n=limit, random_state=seed).reset_index(drop=True)
    return out


class SDFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sdf_dir: str, targets: List[str], y_mean=None, y_std=None):
        self.df = df.reset_index(drop=True)
        self.sdf_dir = sdf_dir
        self.targets = targets
        self.y = self.df[targets].values.astype(np.float32)
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(os.path.join(self.sdf_dir, f"{row['name']}.npy")).astype(np.float32)
        y = self.y[idx]
        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y), row["name"]


class Small3DCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return self.head(x)


def split_dataframe(df: pd.DataFrame, seed: int, test_ratio: float, val_ratio: float):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    test_df = df.iloc[:n_test].reset_index(drop=True)
    val_df = df.iloc[n_test:n_test + n_val].reset_index(drop=True)
    train_df = df.iloc[n_test + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def evaluate(model, loader, device, y_mean, y_std, targets):
    model.eval()
    names, ys, preds = [], [], []
    with torch.no_grad():
        for x, y, batch_names in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y = y.numpy()
            pred = pred * y_std + y_mean
            y = y * y_std + y_mean
            names.extend(batch_names)
            ys.append(y)
            preds.append(pred)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    metrics = []
    for idx, target in enumerate(targets):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        spearman = float(pd.Series(yp).corr(pd.Series(yt), method="spearman"))
        metrics.append({"target": target, "mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman})
    pred_df = pd.DataFrame({"name": names})
    for idx, target in enumerate(targets):
        pred_df[f"{target}_true"] = y_true[:, idx]
        pred_df[f"{target}_pred"] = y_pred[:, idx]
        pred_df[f"{target}_abs_error"] = np.abs(y_pred[:, idx] - y_true[:, idx])
    return metrics, pred_df


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = matched_dataframe(args.sdf_dir, args.property_csv, args.targets, args.limit, args.seed)
    train_df, val_df, test_df = split_dataframe(df, args.seed, args.test_ratio, args.val_ratio)

    y_mean = train_df[args.targets].values.astype(np.float32).mean(axis=0)
    y_std = train_df[args.targets].values.astype(np.float32).std(axis=0) + 1e-8

    sample_x = np.load(os.path.join(args.sdf_dir, f"{train_df.iloc[0]['name']}.npy")).astype(np.float32)
    model = Small3DCNN(in_channels=sample_x.shape[0], out_dim=len(args.targets)).to(args.device)

    train_ds = SDFDataset(train_df, args.sdf_dir, args.targets, y_mean, y_std)
    val_ds = SDFDataset(val_df, args.sdf_dir, args.targets, y_mean, y_std)
    test_ds = SDFDataset(test_df, args.sdf_dir, args.targets, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                val_losses.append(loss_fn(model(x), y).item())
        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics, pred_df = evaluate(model, test_loader, args.device, y_mean, y_std, args.targets)

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    torch.save(best_state, os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
