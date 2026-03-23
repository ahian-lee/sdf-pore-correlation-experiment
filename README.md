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

## Notes

- The current run uses the visible `autofusion` SDF subset, not the full archived `250k` SDF store.
- The `5000`-sample run is a subset experiment intended to validate the signal before scaling up further.
