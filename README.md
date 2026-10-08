# neurosecurity_stack

Alpha-band injection attack validation for a privacy-preserving neurosecurity stack: DP federated learning, a tamper-evident signature, and an anomaly detector, plus passive inference and active poisoning. Eight cohorts from six public datasets: endogenous (Cho, Lee), exogenous (Won, Zhang), no-driver rest (Wang, COG-BCI; eyes-open and eyes-closed each).

## Layout

| File | What it does |
|---|---|
| `run_simulation.py` | full run over all cohorts: signature+detection sweep, private training, passive inference; writes `results/<cohort>.json` |
| `raw_loader.py` | per-dataset raw EEG loaders (Cho, Lee, Won, Zhang, Wang, COG-BCI) and the clean (ICA/CAR/interpolation) variants |
| `preprocess.py` | the clean-loader pipeline: IIR bandpass, resample, bad-channel interpolation, CAR, ICA |
| `cohort.py` | subject sampling, train/calib/test split, client construction |
| `config.py` | every constant: paths, sigma grid, LSH/privacy/federated-training parameters, subject seeds |
| `utils.py` | deterministic seeding (`seed_from`) and JSON load/save, shared by `analyses/*` |
| `features.py` | alpha-band log-power encoder |
| `attack.py` | sensor-level sinusoidal injection |
| `privacy.py` | clipping, Gaussian mechanism, RDP accounting |
| `detect.py` | baseline fitting, Mahalanobis/L2 anomaly score, EMA |
| `signature.py` | random projection, LSH fingerprint, Ed25519 signing |
| `model.py` | logistic regression: loss, gradient, accuracy |
| `fedprox.py` | DP-FedProx-EEG: local DP-SGD, federated averaging, convergence theorem bound |
| `inversion.py` | gradient-inversion reconstruction attack |
| `analyses/` | checks behind specific paper claims (detection theorem, convergence theorem, poisoning sweeps, autocorrelation, joint adversary); see its own README |
| `plots/colors.py` | shared per-cohort colors and axis labels |
| `plots/plot_results.py` | detection, signature, training-integrity, passive-inference figures |
| `plots/plot_convergence.py` | convergence-vs-convergence-theorem-bound figure |
| `plots/plot_subsample_sensitivity.py` | subsample-sensitivity figure |
| `results/`, `results_clean/` | JSON outputs and figures; see `results/README.md` |

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
python -m plots.plot_convergence
python -m analyses.bound_check cho
```

`--quick` uses 4 subjects and 6 rounds. `--cohorts` runs a subset, for when only some datasets are downloaded.
