# Grokking and double descent in XGBoost

This repository holds the code and first results of my research project with Ms. Emily. The project asks three questions about XGBoost: when do grokking and double descent happen, why do they happen, and what do they mean for early stopping? The full plan is in [`docs/research_proposal.pdf`](docs/research_proposal.pdf).

- **Grokking:** the model memorizes its training data first and only much later starts to do well on new data.
- **Double descent over rounds:** as trees are added, test error goes down, then up, then down again.

## Stage 1a: do they happen at all?

Before the full study, Ms. Emily asked me to check whether grokking and double descent happen in XGBoost at all, and whether there is anything like the feature learning of neural networks. Stage 1a answers this with 580 XGBoost models, trained on 22 November 2026.

The results, with every table and figure explained, are in [`analysis_stage1a.ipynb`](analysis_stage1a.ipynb). In short:

- **Grokking:** yes, but rarely, and mostly with few training rows and the brakes off (`min_child_weight` and `lambda` set to 0).
- **Double descent:** yes, but small, and mainly on synthetic data with flipped labels. On real data, test accuracy peaks early and then falls.
- **Feature learning:** no sign of it on modular addition.

## What is in this repository

| Path | What it is |
|---|---|
| `xgbgrok.py` | All experiment code: data, runs, detectors and summary tables |
| `run_stage1a.ipynb` | Colab notebook that trains the Stage 1a models |
| `analysis_stage1a.ipynb` | Results notebook: tables, figures and what they show |
| `docs/research_proposal.pdf` | The research proposal |
| `docs/stage1a_runs.md` | How the Stage 1a runs were designed |
| `results/summary/` | `stage1a_runs.csv` (one row per run) and `stage1a_table.csv` (counts per task and setting) |
| `figures/stage1a_analysis/` | Figures from the results notebook |
| `data/raw/*.json` | Which OpenML dataset versions were used, with checksums |

**Not in this repository:** the raw output of each run (curves at every round, tree statistics, the trained model and the data indices). It is thousands of files, so it stays on Google Drive. The real datasets are not stored here either; the code downloads them from OpenML.

## How to rerun

1. Copy this repository into a Google Drive folder called `MyDrive/xgb-grokking`.
2. Open `run_stage1a.ipynb` in Colab and run cells 1 to 7. Cell 2 installs XGBoost 3.4.1. The code needs exactly this version, because newer versions compute the multiclass hessian differently. The 580 runs took about 2 hours on Colab.
3. Open `analysis_stage1a.ipynb` and run all cells.

Each finished run is saved under an ID made from its settings, so if Colab disconnects, running the notebook again skips the runs that are already done.

To run outside Colab: `pip install -r requirements.txt`.

## Status

- The code also defines 10 MNIST runs, which I have not run yet.
- SDSS17 was run but is left out of the results. Some of its columns work almost like ID numbers, so the model fits its training labels after one round.
- The grokking and double descent detectors are a first version. They only read the saved curves, so they can change without retraining.
