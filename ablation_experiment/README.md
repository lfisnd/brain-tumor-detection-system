# Ablation Experiment

This folder contains a lightweight ablation evaluation script and example result files.

## Files

| File | Description |
| --- | --- |
| `ablation_evaluation.py` | Runs the four experiment groups on the validation set. |
| `ablation_results.csv` | Example tabular metrics. |
| `ablation_results.json` | Example detailed metrics. |

## Experiment Groups

| Group | Configuration |
| --- | --- |
| A | Classification model only |
| B | Segmentation model only |
| C | Classification + segmentation fusion |
| D | Combined model |

## Run

From the project root:

```bash
python ablation_experiment/ablation_evaluation.py
```

The script expects model weights under `weights/` and validation data under `datasets/`.
