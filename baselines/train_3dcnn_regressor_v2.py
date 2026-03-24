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
    parser = argparse.ArgumentParser(description="Enhanced 3D CNN regressor on SDF grids.")
    parser.add_argument("--sdf_dir", required=True)
    parser.add_argument("--property_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--targets", nargs="+", default=["PLD", "LCD"])
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
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
        if os.path.exists(os.path.join(sdf_dir, f"{name}.npy")):
            matched.append(name)
    out = df[df["name"].isin(matched)].reset_index(drop=True)
    if limit > 0 and len(out) > limit:
        out = out.sample(n=limit, random_state=seed).reset_index(drop=True)
    return out


def split_dataframe(df: pd.DataFrame, seed: int, test_ratio: float, val_ratio: float):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    test_df = df.iloc[:n_test].reset_index(drop=True)
    val_df = df.iloc[n_test:n_test + n_val].reset_index(drop=True)
    train_df = df.iloc[n_test + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def random_flip_3d(x: np.ndarray) -> np.ndarray:
    for axis in [1, 2, 3]:
        if random.random() < 0.5:
            x = np.flip(x, axis=axis).copy()
    return x


class SDFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sdf_dir: str, targets: List[str], y_mean=None, y_std=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.sdf_dir = sdf_dir
        self.targets = targets
        self.y = self.df[targets].values.astype(np.float32)
        self.y_mean = y_mean
        self.y_std = y_std
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(os.path.join(self.sdf_dir, f"{row['name']}.npy")).astype(np.float32)
        if self.augment:
            x = random_flip_3d(x)
        y = self.y[idx]
        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y), row["name"]


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class Strong3DCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock3D(32, 32, stride=1, dropout=dropout),
            ResidualBlock3D(32, 64, stride=2, dropout=dropout),
            ResidualBlock3D(64, 64, stride=1, dropout=dropout),
            ResidualBlock3D(64, 128, stride=2, dropout=dropout),
            ResidualBlock3D(128, 128, stride=1, dropout=dropout),
            ResidualBlock3D(128, 192, stride=2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


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
    model = Strong3DCNN(in_channels=sample_x.shape[0], out_dim=len(args.targets), dropout=args.dropout).to(args.device)

    train_ds = SDFDataset(train_df, args.sdf_dir, args.targets, y_mean, y_std, augment=True)
    val_ds = SDFDataset(val_df, args.sdf_dir, args.targets, y_mean, y_std, augment=False)
    test_ds = SDFDataset(test_df, args.sdf_dir, args.targets, y_mean, y_std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=args.min_lr
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                val_losses.append(loss_fn(model(x), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr_now:.6e}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    metrics, pred_df = evaluate(model, test_loader, args.device, y_mean, y_std, args.targets)

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    torch.save(best_state, os.path.join(args.output_dir, "model.pt"))
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"best_epoch={best_epoch} best_val={best_val:.6f}")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
