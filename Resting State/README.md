# Resting State Alpha Injection

Python versions of the original Colab resting-state alpha injection notebook,
split by reusable helpers and dataset group.

## Files

- `common.py` - shared preprocessing, injection, scoring, and plotting helpers
- `ds004148_group.py` - OpenNeuro ds004148 EO/EC experiment
- `amplitude_sweep.py` - ds004148 sub-01 amplitude sweep
- `cog_bci_group.py` - COG-BCI EO/EC experiment

## Setup

```bash
python3 -m pip install mne scikit-learn matplotlib pandas numpy
```

## Run

Run from this folder:

```bash
python3 ds004148_group.py
python3 amplitude_sweep.py
python3 cog_bci_group.py
```

Outputs are written to `results/resting_state/`.

