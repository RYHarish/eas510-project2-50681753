# EAS 510 – Project 2: Detecting Windows Malware from Static Features

Classical ML (logistic regression, decision tree, random forest, gradient boosting) on the
**EMBER 2018 v2** pre-extracted static features of Windows PE files, comparing a **random
train/test split** with a **time-based split** (train on older samples, test on newer ones).
Only extracted features are used: no executable is downloaded, handled, or run.

## Project Structure
```
README.md            overview, setup, run instructions
report.md            official final report
main.py              single entry point
requirements.txt     pinned dependencies
config.py            seeds, paths, subsample size, split cutoff, hyperparameter grids
src/
  features.py        raw EMBER JSON -> 2,381-feature vector (own re-implementation, no lief)
  data.py            index raw files, stratified subsample, vectorize, cache, bundled sample
  splits.py          random vs. time split (same pool, same training size)
  eda.py             class balance, missing values, feature scales
  models.py          L1/L2 logistic regression (SGD) + decision tree, with CV grids
  evaluate.py        ROC-AUC, precision/recall/F1, confusion matrix, recall @ 0.1%/1% FPR
  experiment.py      CV tuning on the training period, test evaluation, plots
notebooks/eda.ipynb  interactive look at the raw JSON columns
data/
  sample/ember_sample.npz   bundled runnable sample (1,999 vectorized rows)
  download.py               downloads + extracts full EMBER 2018 into data/raw/
results/
  sample/            results of `python main.py`
  full/              results reported in report.md (150k-row subsample)
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.14.

## Run
```bash
# 1) Bundled sample: runs everything in well under a minute, no download needed
python main.py

# 2) Full experiment (~10 min on a 14-core laptop, ~11 GB disk)
python data/download.py                 # 1.6 GB download -> data/raw/ember2018/
python main.py --data full              # builds data/processed/ on first run, then EDA + baselines

# run single stages
python main.py --data full --stage prepare     # subsample + vectorize + splits only
python main.py --data full --stage eda
python main.py --data full --stage baselines
```
Outputs go to `results/<sample|full>/` (`splits.csv`, `eda/`, `baselines/`).

## Data and subsampling
- Pool: the 750,000 labeled EMBER 2018 samples first seen Jan–Dec 2018. The 50,000 benign-only
  pre-2018 files are excluded, so both splits draw from the same period.
- Subsample: 149,999 rows (20%), drawn at random within every (month × label) cell with
  `SEED = 42`. The exact rows are listed in `data/processed/meta.csv`, which is rebuilt
  deterministically by `main.py`.
- Time split: train = Jan–Jun 2018 (60,794 rows), test = Jul–Dec 2018 (89,205 rows).
- Random split: stratified random split of the same pool with the same 60,794 training rows.
- Hyperparameters are tuned with 3-fold CV on the training rows only.

## Reproducibility
Seeds, subsample size, split cutoff and grids are in `config.py`. The sample run
reproduces `results/sample/` exactly.
