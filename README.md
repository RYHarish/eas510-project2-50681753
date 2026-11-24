# EAS 510 – Project 2

## Overview
_Brief description of the problem, dataset, and approach._

## Project Structure
```
README.md          overview, setup, run instructions
report.md          official final report
main.py            single entry point
requirements.txt   pinned dependencies
config.py          seeds, settings, hyperparameters
src/               data processing and model code
data/sample/       bundled, runnable sample
data/download.py   optional full-data download script
results/sample/    documented sample-run results
results/full/      report results, when different
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# Run on the bundled sample (default)
python main.py

# Run on the full dataset (download first)
python data/download.py
python main.py --data full
```

Outputs are written to `results/sample/` or `results/full/`.

## Reproducibility
All seeds and hyperparameters are set in `config.py`.
