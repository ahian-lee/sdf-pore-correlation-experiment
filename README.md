# SDF-Pore Correlation Experiment

This repository packages a focused experiment testing whether MOF SDF fields correlate with pore descriptors such as `PLD`, `LCD`, `VF`, and `GCD`.

## Data Used

- SDF NPY directory:
  `/opt/data/private/moffusion/autofusion/data/resolution_32`
- Property table:
  `/opt/data/private/moffusion/data/properties/250k_pld_processed.csv`
- CIF root used to establish the paired dataset:
  `/opt/data/private/moffusion/250k_cif`

Current experiment runs on the `autofusion` SDF subset, using `5000` matched samples.

## Files

- `experiment_sdf_pore_correlation.py`
  Main analysis script for feature extraction, correlation analysis, and baseline regression.
- `plot_sdf_pore_results.py`
  Plotting script for correlation and performance figures.
- `plot_prediction_diagnostics.py`
  Generates true-vs-predicted scatter plots and binned error plots for `PLD/LCD`.
- `baselines/`
  Server-oriented baseline scripts for a raw `3D CNN` model and a CIF-derived graph model.
- `results/`
  CSV and JSON outputs from the `limit=5000` run.
- `figures/`
  PNG figures generated from the `limit=5000` run.

## Main Result

On the `5000`-sample experiment:

- `PLD`
  - `ExtraTrees`: `R2 = 0.8069`, `MAE = 0.5763`, `Spearman = 0.8748`
- `LCD`
  - `Ridge`: `R2 = 0.9554`, `MAE = 0.4535`, `Spearman = 0.9724`
  - `ExtraTrees`: `R2 = 0.9524`, `MAE = 0.4525`, `Spearman = 0.9697`
- `VF`
  - `ExtraTrees`: `R2 = 0.9877`, `MAE = 0.0081`, `Spearman = 0.9951`
- `GCD`
  - `ExtraTrees`: `R2 = 0.9925`, `MAE = 0.1781`, `Spearman = 0.9951`

These results support a strong mapping from SDF distribution statistics to MOF pore descriptors, with:

- `PLD` tracking high-quantile channel statistics such as `ch0_q95` and `ch0_q75`
- `LCD` tracking cavity-scale features such as `ch0_max` and `ch0_q95`

## Direct PLD Proxy From SDF

We also tested a geometry-only `PLD proxy` computed directly from the `32^3` SDF without any learned model.

Definition:

- Use `channel 0` of the SDF
- Threshold the SDF at radius `r`
- Treat voxels with `SDF >= r` as probe-center accessible
- Check whether a periodic percolating path still exists
- Take the largest such `r`, and define `PLD proxy = 2r`

This matches the `pore limiting diameter` definition much better than global statistics like `max(SDF)`.

On a `200`-sample matched subset:

- Raw proxy:
  - `MAE = 0.797 A`
  - `RMSE = 0.839 A`
  - `Bias = -0.797 A`
  - `Pearson = 0.991`
  - `Spearman = 0.988`
- After a simple linear calibration:
  - `MAE = 0.196 A`
  - `RMSE = 0.256 A`
  - `Bias ~ 0`

These results show that even at `32^3`, the direct SDF-derived bottleneck quantity strongly reflects the real `PLD`, though it systematically underestimates the value before calibration.

## Reproduce

Run the analysis:

```bash
python experiment_sdf_pore_correlation.py \
  --limit 5000 \
  --output_dir results
```

Generate figures:

```bash
python plot_sdf_pore_results.py \
  --input_dir results \
  --output_dir figures
```

Prediction diagnostics:

```bash
python plot_prediction_diagnostics.py \
  --input_csv results/merged_sdf_properties.csv \
  --output_dir diagnostics \
  --model extra_trees
```

Direct PLD proxy analysis:

```bash
python proxy_analysis/compute_periodic_pld_proxy.py \
  --limit 200 \
  --output-dir proxy_analysis/results_ch0_200

python proxy_analysis/analyze_periodic_pld_proxy.py \
  --results-csv proxy_analysis/results_ch0_200/pld_proxy_results.csv \
  --output-dir proxy_analysis/results_ch0_200/analysis
```

## Server Baselines

Raw `3D CNN` baseline:

```bash
bash baselines/run_3dcnn_baseline.sh
```

Graph baseline from CIF:

```bash
bash baselines/run_graph_baseline.sh
```

Both scripts support environment-variable overrides. Example:

```bash
LIMIT=10000 DEVICE=cuda OUTPUT_DIR=./baseline_runs/3dcnn_10k \
  bash baselines/run_3dcnn_baseline.sh
```

## Notes

- The current run uses the visible `autofusion` SDF subset, not the full archived `250k` SDF store.
- The `5000`-sample run is a subset experiment intended to validate the signal before scaling up further.
