# neurosecurity_stack

Simulation code for the alpha-band injection attack validation.
Tests all three pillars (DP-FL, signaturing, anomaly detection) plus passive
inference and active poisoning, on four real EEG datasets (Cho, Lee, Won,
Zhang).

## Layout

- `raw_loader.py` - loads and preprocesses raw EEG from each dataset
- `features.py` - alpha-band power encoder
- `attack.py` - the sensor-level injection attack
- `privacy.py` - Gaussian DP mechanism, RDP accounting
- `detect.py` - Pillar 3 baseline and anomaly scoring
- `signature.py` - Pillar 2 projection, LSH, Ed25519 signing
- `model.py` - logistic regression used by Pillar 1
- `autocorrelation.py` - corrected clipping norm
- `fedprox.py` - Pillar 1 DP-FedProx-EEG training
- `inversion.py` - passive inference (gradient reconstruction) attack
- `run_simulation.py` - runs everything, writes `results/<dataset>.json`
- `plots.py` - builds the figures in `results/figures/`
- `Resting State/` - resting-state EO/EC alpha injection scripts split by dataset group

## Running

```
python3 run_simulation.py --quick   # small smoke test
python3 run_simulation.py           # full run, all 4 datasets
python3 plots.py                    # regenerate figures from results/
```

Raw data is expected under `./data/` by default (Cho, Lee, won_raw, zhang_raw),
or set `EEG_RAW_ROOT` to point elsewhere. 

Needs numpy, scipy, mne, h5py, cryptography, matplotlib.
