import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute a periodic PLD proxy from 32^3 SDF volumes and compare against true PLD."
    )
    parser.add_argument(
        "--sdf-dir",
        type=Path,
        default=Path("/opt/data/private/moffusion/autofusion/data/resolution_32"),
        help="Directory containing SDF .npy files.",
    )
    parser.add_argument(
        "--properties-csv",
        type=Path,
        default=Path("/opt/data/private/moffusion/data/properties/250k_pld_processed.csv"),
        help="CSV with columns name and PLD.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="SDF channel to use for the PLD proxy.",
    )
    parser.add_argument(
        "--scale-angstrom-per-unit",
        type=float,
        default=30.0,
        help="Convert normalized SDF units back to Angstrom. xyz_to_periodic_sdf.py implies 30 A per unit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of matched samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling matched files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/data/private/moffusion/outputs/periodic_pld_proxy_eval"),
        help="Directory to save CSV and JSON outputs.",
    )
    return parser.parse_args()


def load_matches(sdf_dir: Path, properties_csv: Path) -> pd.DataFrame:
    props = pd.read_csv(properties_csv, usecols=["name", "PLD"]).dropna(subset=["PLD"])
    available = {path.stem for path in sdf_dir.glob("*.npy") if not path.name.endswith("_occ.npy")}
    matched = props[props["name"].isin(available)].copy()
    matched["sdf_path"] = matched["name"].map(lambda name: str(sdf_dir / f"{name}.npy"))
    matched = matched.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return matched


def periodic_neighbors(index, shape):
    x, y, z = index
    nx, ny, nz = shape
    for axis, delta in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        coords = [x, y, z]
        shift = [0, 0, 0]
        original = coords[axis]
        coords[axis] = (coords[axis] + delta) % shape[axis]
        if delta == 1 and original == shape[axis] - 1:
            shift[axis] = 1
        elif delta == -1 and original == 0:
            shift[axis] = -1
        yield tuple(coords), tuple(shift)


def has_periodic_path(mask: np.ndarray) -> bool:
    shape = mask.shape
    visited = np.zeros(shape, dtype=bool)
    displacements = np.zeros(shape + (3,), dtype=np.int16)

    open_points = np.argwhere(mask)
    for sx, sy, sz in open_points:
        start = (int(sx), int(sy), int(sz))
        if visited[start]:
            continue

        visited[start] = True
        displacements[start] = (0, 0, 0)
        queue = deque([start])

        while queue:
            node = queue.popleft()
            base_disp = displacements[node]
            for neighbor, wrap_shift in periodic_neighbors(node, shape):
                if not mask[neighbor]:
                    continue

                target_disp = base_disp + np.array(wrap_shift, dtype=np.int16)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    displacements[neighbor] = target_disp
                    queue.append(neighbor)
                    continue

                cycle = target_disp - displacements[neighbor]
                if np.any(cycle != 0):
                    return True

    return False


def pld_proxy_from_sdf(sdf_channel: np.ndarray, scale_angstrom_per_unit: float) -> float:
    radii = np.asarray(sdf_channel, dtype=np.float32) * scale_angstrom_per_unit
    positive = np.unique(radii[radii > 0])
    if positive.size == 0:
        return 0.0

    left = 0
    right = positive.size - 1
    best = 0.0

    while left <= right:
        mid = (left + right) // 2
        threshold = float(positive[mid])
        mask = radii >= threshold
        if has_periodic_path(mask):
            best = threshold
            left = mid + 1
        else:
            right = mid - 1

    return 2.0 * best


def evaluate(df: pd.DataFrame, channel: int, scale_angstrom_per_unit: float) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        array = np.load(row.sdf_path)
        sdf_channel = array[channel]
        proxy = pld_proxy_from_sdf(sdf_channel, scale_angstrom_per_unit)
        true_pld = float(row.PLD)
        rows.append(
            {
                "name": row.name,
                "true_pld": true_pld,
                "proxy_pld": proxy,
                "abs_error": abs(proxy - true_pld),
                "signed_error": proxy - true_pld,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    diff = results["proxy_pld"] - results["true_pld"]
    summary = {
        "num_samples": int(len(results)),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "bias": float(diff.mean()),
        "pearson": float(results[["proxy_pld", "true_pld"]].corr(method="pearson").iloc[0, 1]),
        "spearman": float(results[["proxy_pld", "true_pld"]].corr(method="spearman").iloc[0, 1]),
        "true_pld_min": float(results["true_pld"].min()),
        "true_pld_max": float(results["true_pld"].max()),
        "proxy_pld_min": float(results["proxy_pld"].min()),
        "proxy_pld_max": float(results["proxy_pld"].max()),
    }
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched = load_matches(args.sdf_dir, args.properties_csv)
    matched = matched.sample(n=min(args.limit, len(matched)), random_state=args.seed).reset_index(drop=True)

    results = evaluate(matched, args.channel, args.scale_angstrom_per_unit)
    summary = summarize(results)

    results_path = args.output_dir / "pld_proxy_results.csv"
    summary_path = args.output_dir / "summary.json"
    results.to_csv(results_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {results_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
