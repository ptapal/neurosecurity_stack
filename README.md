# neurosecurity_stack

Alpha-band injection attack validation. Three pillars (DP-FL, signaturing,
anomaly detection), plus passive inference and active poisoning. Eight
cohorts: endogenous (Cho, Lee), exogenous (Won, Zhang), no-driver rest
(Wang, COG-BCI; EO/EC each).

## Layout

- `raw_loader.py` - raw EEG loading, preprocessing
- `features.py` - alpha-band encoder
- `attack.py` - sensor-level injection
- `privacy.py` - Gaussian mechanism, RDP
- `detect.py` - Pillar 3 baseline, scoring
- `signature.py` - Pillar 2 projection, LSH, Ed25519
- `model.py` - Pillar 1 logistic regression
- `autocorrelation.py` - corrected clipping norm
- `fedprox.py` - Pillar 1 DP-FedProx-EEG
- `inversion.py` - passive inference attack
- `run_simulation.py` - full pipeline, writes `results/<cohort>.json`
- `plots.py` - figures from `results/`

## Running

```
python3 run_simulation.py --quick
python3 run_simulation.py
python3 plots.py
```

Raw data under `./data/` (Cho, Lee, won_raw, zhang_raw, wang_raw, cogbci_raw),
or set `EEG_RAW_ROOT`.

numpy, scipy, mne, h5py, cryptography, matplotlib.
