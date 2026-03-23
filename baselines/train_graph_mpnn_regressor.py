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
from ase.io import read
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple graph regressor on CIF-derived graphs.")
    parser.add_argument("--cif_dir", required=True)
    parser.add_argument("--property_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--targets", nargs="+", default=["PLD", "LCD"])
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=3)
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


def matched_dataframe(cif_dir: str, property_csv: str, targets: List[str], limit: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(property_csv, usecols=["name"] + targets).dropna().drop_duplicates("name")
    matched = []
    for name in df["name"]:
        if os.path.exists(os.path.join(cif_dir, f"{name}.cif")):
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


def atoms_to_graph(cif_path: str, cutoff: float):
    atoms = read(cif_path)
    z = np.array(atoms.get_atomic_numbers(), dtype=np.int64)
    pos = atoms.get_positions().astype(np.float32)
    i_idx, j_idx, offsets = neighbor_list("ijS", atoms, cutoff)
    cell = atoms.cell.array.astype(np.float32)
    disp = pos[j_idx] + offsets @ cell - pos[i_idx]
    dist = np.linalg.norm(disp, axis=1, keepdims=True).astype(np.float32)
    edge_index = np.stack([i_idx, j_idx], axis=0).astype(np.int64)
    return {
        "z": torch.from_numpy(z),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(dist),
        "n_nodes": len(z),
    }


class GraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cif_dir: str, targets: List[str], cutoff: float, y_mean=None, y_std=None):
        self.df = df.reset_index(drop=True)
        self.cif_dir = cif_dir
        self.targets = targets
        self.y = self.df[targets].values.astype(np.float32)
        self.y_mean = y_mean
        self.y_std = y_std
        self.cutoff = cutoff
        self.graphs = []
        for _, row in self.df.iterrows():
            graph = atoms_to_graph(os.path.join(self.cif_dir, f"{row['name']}.cif"), self.cutoff)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = self.y[idx]
        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std
        return self.graphs[idx], torch.from_numpy(y), self.df.iloc[idx]["name"]


def collate_graphs(batch):
    zs, edge_indices, edge_attrs, batch_idx, ys, names = [], [], [], [], [], []
    node_offset = 0
    for graph_id, (graph, y, name) in enumerate(batch):
        n = graph["n_nodes"]
        zs.append(graph["z"])
        edge_indices.append(graph["edge_index"] + node_offset)
        edge_attrs.append(graph["edge_attr"])
        batch_idx.append(torch.full((n,), graph_id, dtype=torch.long))
        ys.append(y)
        names.append(name)
        node_offset += n
    return {
        "z": torch.cat(zs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "y": torch.stack(ys, dim=0),
        "names": names,
    }


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        m_in = torch.cat([h[src], h[dst], edge_attr], dim=1)
        msg = self.msg(m_in)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        upd_in = torch.cat([h, agg], dim=1)
        return h + self.upd(upd_in)


class SimpleMPNN(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(101, hidden_dim)
        self.layers = nn.ModuleList([MessageLayer(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, edge_index, edge_attr, batch):
        h = self.embed(z)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        n_graphs = int(batch.max().item()) + 1
        pooled = torch.zeros((n_graphs, h.size(1)), device=h.device, dtype=h.dtype)
        counts = torch.zeros((n_graphs, 1), device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype))
        pooled = pooled / counts.clamp_min(1.0)
        return self.head(pooled)


def evaluate(model, loader, device, y_mean, y_std, targets):
    model.eval()
    ys, preds, names = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["z"].to(device),
                batch["edge_index"].to(device),
                batch["edge_attr"].to(device),
                batch["batch"].to(device),
            ).cpu().numpy()
            y = batch["y"].numpy()
            pred = pred * y_std + y_mean
            y = y * y_std + y_mean
            ys.append(y)
            preds.append(pred)
            names.extend(batch["names"])
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

    df = matched_dataframe(args.cif_dir, args.property_csv, args.targets, args.limit, args.seed)
    train_df, val_df, test_df = split_dataframe(df, args.seed, args.test_ratio, args.val_ratio)

    y_mean = train_df[args.targets].values.astype(np.float32).mean(axis=0)
    y_std = train_df[args.targets].values.astype(np.float32).std(axis=0) + 1e-8

    train_ds = GraphDataset(train_df, args.cif_dir, args.targets, args.cutoff, y_mean, y_std)
    val_ds = GraphDataset(val_df, args.cif_dir, args.targets, args.cutoff, y_mean, y_std)
    test_ds = GraphDataset(test_df, args.cif_dir, args.targets, args.cutoff, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_graphs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_graphs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_graphs)

    model = SimpleMPNN(args.hidden_dim, len(args.targets), args.num_layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            pred = model(
                batch["z"].to(args.device),
                batch["edge_index"].to(args.device),
                batch["edge_attr"].to(args.device),
                batch["batch"].to(args.device),
            )
            y = batch["y"].to(args.device)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["z"].to(args.device),
                    batch["edge_index"].to(args.device),
                    batch["edge_attr"].to(args.device),
                    batch["batch"].to(args.device),
                )
                y = batch["y"].to(args.device)
                val_losses.append(loss_fn(pred, y).item())
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
