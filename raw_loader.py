import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import mne
import numpy as np
import scipy.io
from scipy.signal import resample_poly

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mne")

ROOT = Path(os.environ.get("EEG_RAW_ROOT", "./data"))

L_FREQ = 1.0
H_FREQ = 40.0
FS_OUT = 250.0

CH32 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]

CHO_CH = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
CHO_N_EEG = 64
CHO_FS = 512
CHO_IMG_SAMPS = 2048

LEE_CH = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3",
    "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
    "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz",
    "CP4", "P1", "P2", "POz", "FT9", "FTT9h", "TTP7h", "TP7",
    "TPP9h", "FT10", "FTT10h", "TPP8h", "TP8", "TPP10h", "F9", "F10",
    "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
]
LEE_FS = 1000
LEE_SAMPS = 4000

WON_MAP = {
    "FP1": "Fp1", "AF3": "AF3", "F7": "F7", "F3": "F3",
    "FC1": "FC1", "FC5": "FC5", "T7": "T7", "C3": "C3",
    "CP1": "CP1", "CP5": "CP5", "P7": "P7", "P3": "P3",
    "PZ": "Pz", "PO3": "PO3", "O1": "O1", "OZ": "Oz",
    "O2": "O2", "PO4": "PO4", "P4": "P4", "P8": "P8",
    "CP6": "CP6", "CP2": "CP2", "C4": "C4", "T8": "T8",
    "FC6": "FC6", "FC2": "FC2", "F4": "F4", "F8": "F8",
    "AF4": "AF4", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz",
}
WON_PRE, WON_POST = 0.1, 0.7

ZHANG_CH = [
    "Fpz", "Fp1", "Fp2", "Af3", "Af4", "Af7", "Af8", "Fz", "F1", "F2", "F3", "F4",
    "F5", "F6", "F7", "F8", "Fcz", "Fc1", "Fc2", "Fc3", "Fc4", "Fc5", "Fc6", "Ft7",
    "Ft8", "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", "Cp1", "Cp2",
    "Cp3", "Cp4", "Cp5", "Cp6", "Tp7", "Tp8", "Pz", "P3", "P4", "P5", "P6", "P7",
    "P8", "Poz", "Po3", "Po4", "Po7", "Po8", "Oz", "O1", "O2",
]
ZHANG_RENAME = {
    "Af3": "AF3", "Af4": "AF4", "Fc1": "FC1", "Fc2": "FC2", "Fc5": "FC5",
    "Fc6": "FC6", "Cp1": "CP1", "Cp2": "CP2", "Cp5": "CP5", "Cp6": "CP6",
    "Po3": "PO3", "Po4": "PO4",
}
ZHANG_FS = 1000.0
ZHANG_PRE, ZHANG_POST = 0.1, 0.7
# code 2 is the rare target, code 1 is common non-target (checked against real trigger counts)
ZHANG_TARGET = 2
ZHANG_NONTARGET = 1


@dataclass
class SubjectData:
    subject_id: str
    paradigm: str
    trials: np.ndarray
    labels: np.ndarray


def _filt(raw):
    raw.pick_channels(CH32, ordered=True)
    raw.filter(L_FREQ, H_FREQ, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    raw.resample(FS_OUT, verbose=False)
    return raw


def load_cho_subject(subject_id, max_trials=200):
    path = ROOT / "Cho" / "gigadb-datasets" / "live" / "pub" / "10.5524" / \
        "100001_101000" / "100295" / "mat_data" / f"{subject_id}.mat"
    mat = scipy.io.loadmat(str(path), simplify_cells=True)
    eeg = mat["eeg"]
    n_trials = int(eeg["n_imagery_trials"])
    frame = np.array(eeg["frame"], dtype=np.int32)
    pre = int(abs(int(frame[0])) * CHO_FS // 1000)

    trials, labels = [], []
    for cls, key in enumerate(("imagery_left", "imagery_right")):
        data = np.array(eeg[key], dtype=np.float64)[:CHO_N_EEG]
        info = mne.create_info(ch_names=CHO_CH, sfreq=CHO_FS, ch_types="eeg")
        raw = _filt(mne.io.RawArray(data, info, verbose=False))
        d = raw.get_data()

        total = data.shape[1] // n_trials
        total_r = round(total * FS_OUT / CHO_FS)
        pre_r = round(pre * FS_OUT / CHO_FS)
        img_r = round(CHO_IMG_SAMPS * FS_OUT / CHO_FS)

        ep = d.reshape(len(CH32), n_trials, total_r)
        trials.append(ep[:, :, pre_r:pre_r + img_r])
        labels.append(np.full(n_trials, cls, dtype=np.int64))

    pooled = np.concatenate(trials, axis=1).transpose(1, 0, 2)
    y = np.concatenate(labels)
    if max_trials is not None:
        pooled, y = pooled[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "cho", pooled.astype(np.float32), y)


def _lee_path(session, idx):
    sess = session[-1]
    return (ROOT / "Lee" / "gigadb-datasets" / "live" / "pub" / "10.5524" /
            "100001_101000" / "100542" / session / f"s{idx}" /
            f"sess0{sess}_subj{idx:02d}_EEG_MI.mat")


def load_lee_subject(subject_id, session="session1", max_trials=200):
    idx = int(subject_id.split("-")[-1])
    mat = scipy.io.loadmat(str(_lee_path(session, idx)), simplify_cells=True)

    parts, ys = [], []
    for key in ("EEG_MI_train", "EEG_MI_test"):
        smt = np.array(mat[key]["smt"], dtype=np.float64)
        parts.append(smt.transpose(2, 1, 0))
        y_dec = np.array(mat[key]["y_dec"], dtype=np.int64).ravel()
        ys.append(y_dec - 1)
    pooled = np.concatenate(parts, axis=1)
    y = np.concatenate(ys)
    n_trials = pooled.shape[1]
    flat = pooled.reshape(len(LEE_CH), n_trials * LEE_SAMPS)

    info = mne.create_info(ch_names=LEE_CH, sfreq=LEE_FS, ch_types="eeg")
    raw = _filt(mne.io.RawArray(flat, info, verbose=False))
    d = raw.get_data()

    samps = round(LEE_SAMPS * FS_OUT / LEE_FS)
    ep = d.reshape(len(CH32), n_trials, samps).transpose(1, 0, 2)
    if max_trials is not None:
        ep, y = ep[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "lee", ep.astype(np.float32), y)


def _filt_won(raw):
    raw.pick_channels(CH32, ordered=True)
    raw.filter(L_FREQ, H_FREQ, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    return raw


def load_won_subject(subject_id, max_trials=150, seed=0):
    sub_dir = ROOT / "won_raw" / "eeg_bids" / "Won2022_BIDS" / subject_id
    set_file = sub_dir / "eeg" / f"{subject_id}_task-RSVPtask_run-3_eeg.set"
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    raw.rename_channels({k: v for k, v in WON_MAP.items() if k in raw.ch_names})
    raw = _filt_won(raw)

    events, _ = mne.events_from_annotations(raw, verbose=False)
    tgt = events[events[:, 2] == 1]
    ntg = events[events[:, 2] == 2]
    if len(tgt) == 0:
        raise ValueError(f"no target events for {subject_id}")

    rng = np.random.default_rng(seed)
    n_keep = min(len(ntg), len(tgt) * 2)
    ntg = ntg[rng.choice(len(ntg), size=n_keep, replace=False)]

    all_ev = np.concatenate([tgt, ntg], axis=0)
    all_ev = all_ev[np.argsort(all_ev[:, 0])]

    ep = mne.Epochs(raw, all_ev, event_id={"target": 1, "nontarget": 2},
                     tmin=-WON_PRE, tmax=WON_POST, baseline=(-WON_PRE, 0.0),
                     preload=True, verbose=False, event_repeated="drop")
    ep.resample(FS_OUT, verbose=False)
    data = ep.get_data() * 1e6
    y = (ep.events[:, 2] == 1).astype(np.int64)

    if max_trials is not None:
        data, y = data[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "won", data.astype(np.float32), y)


def load_zhang_subject(subject_id, day="Day_1", max_trials=150, seed=0):
    path = ROOT / "zhang_raw" / f"{subject_id}.mat"
    trials, labels = [], []
    pre = int(round(ZHANG_PRE * ZHANG_FS))
    post = int(round(ZHANG_POST * ZHANG_FS))

    with h5py.File(str(path), "r") as f:
        for ref in f[day][:, 0]:
            mat = f[ref][:]
            eeg = mat[:, :57].T.astype(np.float64)
            trig = mat[:, 57]

            info = mne.create_info(ch_names=ZHANG_CH, sfreq=ZHANG_FS, ch_types="eeg")
            raw = mne.io.RawArray(eeg, info, verbose=False)
            raw.rename_channels(ZHANG_RENAME)
            raw.pick_channels(CH32, ordered=True)
            raw.filter(L_FREQ, H_FREQ, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
            d = raw.get_data()

            for code, lab in ((ZHANG_TARGET, 1), (ZHANG_NONTARGET, 0)):
                for pos in np.where(trig == code)[0]:
                    start, end = pos - pre, pos + post
                    if start < 0 or end > d.shape[1]:
                        continue
                    trials.append(d[:, start:end])
                    labels.append(lab)

    ep = np.stack(trials)
    y = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    tgt_idx = np.where(y == 1)[0]
    ntg_idx = np.where(y == 0)[0]
    n_keep = min(len(ntg_idx), len(tgt_idx) * 2)
    ntg_idx = rng.choice(ntg_idx, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([tgt_idx, ntg_idx]))
    ep, y = ep[keep], y[keep]

    ep_rs = resample_poly(ep, up=1, down=4, axis=-1)
    if max_trials is not None:
        ep_rs, y = ep_rs[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "zhang", ep_rs.astype(np.float32), y)


def list_cho_subjects():
    d = ROOT / "Cho" / "gigadb-datasets" / "live" / "pub" / "10.5524" / "100001_101000" / "100295" / "mat_data"
    return sorted(p.stem for p in d.glob("s*.mat"))


def list_lee_subjects(session="session1"):
    d = ROOT / "Lee" / "gigadb-datasets" / "live" / "pub" / "10.5524" / "100001_101000" / "100542" / session
    ids = sorted((p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("s")),
                 key=lambda s: int(s[1:]))
    return [f"sub-{int(i[1:]):02d}" for i in ids]


def list_won_subjects():
    d = ROOT / "won_raw" / "eeg_bids" / "Won2022_BIDS"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def list_zhang_subjects():
    d = ROOT / "zhang_raw"
    return sorted((p.stem for p in d.glob("S*.mat")), key=lambda s: int(s[1:]))
