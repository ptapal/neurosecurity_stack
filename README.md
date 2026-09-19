# neurosecurity_stack

Alpha-band injection attack validation for a three-pillar neurosecurity stack (DP federated learning, private signaturing, anomaly detection), plus passive inference and active poisoning. Eight cohorts from six public datasets: endogenous (Cho, Lee), exogenous (Won, Zhang), no-driver rest (Wang, COG-BCI; eyes-open and eyes-closed each).

## Layout

| Path | Content |
|---|---|
| `run_simulation.py` | full run over all cohorts, writes `results/<cohort>.json` |
| `raw_loader.py`, `cohort.py`, `config.py` | data loading, subject selection, constants |
| `features.py`, `attack.py`, `privacy.py`, `detect.py`, `signature.py`, `model.py`, `fedprox.py`, `inversion.py` | the stack: encoder, alpha injection, DP mechanisms, Pillar 3 detector, Pillar 2 signature, federated training, gradient inversion |
| `analyses/` | paper checks (Theorem 6.1, Theorem 8.1, poisoning sweep, autocorrelation, joint adversary); see its README |
| `plots/` | figure scripts |
| `results/` | JSON outputs and figures |

## Setup

```
pip install -r requirements.txt
```

Python 3.10+. Run everything from the repo root.

## Data

The raw recordings are not in this repository. Download the six public datasets and put each in its own folder under `data/` (or set `EEG_RAW_ROOT` to another folder with the same structure):

| Folder | Dataset | Source |
|---|---|---|
| `data/cho/` | Cho et al. 2017, motor imagery | GigaDB, DOI 10.5524/100295; the `mat_data` files (`s01.mat` ...) |
| `data/lee/` | Lee et al. 2019, OpenBMI | GigaDB, DOI 10.5524/100542; the session 1 motor-imagery files, flat (`sess01_subj01_EEG_MI.mat` ...) |
| `data/won/` | Won et al. 2022, RSVP and P300 speller | figshare, article 19776004; the BIDS archive, with the `sub-*` folders directly in `data/won/` |
| `data/zhang/` | Zhang et al. 2025, longitudinal ERP dataset | figshare, article 27201003; `S1.mat` ... `S15.mat` (about 52 GB) |
| `data/wang/` | Wang et al. 2022, test-retest resting EEG | OpenNeuro, ds004148; the `sub-*` folders directly in `data/wang/` |
| `data/cogbci/` | COG-BCI passive-BCI dataset | Zenodo, record 6874129 (version 1); each `sub-XX.zip` unzipped to `data/cogbci/sub-XX/` |

Each run uses 15 subjects per cohort, drawn with a fixed seed from the subjects present. Download the full datasets, or only the subjects listed in `PAPER_SUBJECTS` in `config.py`; with 15 or fewer subjects present, all of them are used and the results are identical. For COG-BCI only the `RS_*` recordings are read.

## Running

```
python run_simulation.py --quick
python run_simulation.py
python run_simulation.py --cohorts cho lee
python -m plots.plot_results
python -m analyses.bound_check cho
```

`--quick` uses 4 subjects and 6 rounds. `--cohorts` runs a subset, for when only some datasets are downloaded.
