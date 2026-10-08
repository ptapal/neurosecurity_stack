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
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz",
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
LEE_N_EEG = 62

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
# code 2 is the rare target, code 1 is common non-target 
ZHANG_TARGET = 2
ZHANG_NONTARGET = 1


RS_EPOCH_SEC = 2.0


@dataclass
class SubjectData:
    subject_id: str
    paradigm: str
    trials: np.ndarray
    labels: np.ndarray
    times: np.ndarray | None = None
    runs: np.ndarray | None = None
    baseline: np.ndarray | None = None


def _filt(raw):
    raw.pick_channels(CH32, ordered=True)
    raw.filter(L_FREQ, H_FREQ, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    raw.resample(FS_OUT, verbose=False)
    return raw


def load_cho_subject(subject_id, max_trials=200):
    path = ROOT / "cho" / f"{subject_id}.mat"
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
    return ROOT / "lee" / f"sess0{sess}_subj{idx:02d}_EEG_MI.mat"


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
    sub_dir = ROOT / "won" / subject_id
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
    times = ep.events[:, 0] / raw.info["sfreq"]

    if max_trials is not None:
        data, y, times = data[:max_trials], y[:max_trials], times[:max_trials]
    return SubjectData(subject_id, "won", data.astype(np.float32), y,
                       times=times, runs=np.zeros(len(y), dtype=np.int64))


def load_zhang_subject(subject_id, day="Day_1", max_trials=150, seed=0, ordered=False):
    path = ROOT / "zhang" / f"{subject_id}.mat"
    trials, labels, times, runs = [], [], [], []
    pre = int(round(ZHANG_PRE * ZHANG_FS))
    post = int(round(ZHANG_POST * ZHANG_FS))

    with h5py.File(str(path), "r") as f:
        for run, ref in enumerate(f[day][:, 0]):
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
                    times.append(pos)
                    runs.append(run)

    ep = np.stack(trials)
    y = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    tgt_idx = np.where(y == 1)[0]
    ntg_idx = np.where(y == 0)[0]
    n_keep = min(len(ntg_idx), len(tgt_idx) * 2)
    ntg_idx = rng.choice(ntg_idx, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([tgt_idx, ntg_idx]))
    if ordered:
        keep = keep[np.lexsort((np.array(times)[keep], np.array(runs)[keep]))]
    ep, y = ep[keep], y[keep]
    times, runs = np.array(times)[keep] / ZHANG_FS, np.array(runs)[keep]

    ep_rs = resample_poly(ep, up=1, down=4, axis=-1)
    if max_trials is not None:
        ep_rs, y, times, runs = ep_rs[:max_trials], y[:max_trials], times[:max_trials], runs[:max_trials]
    return SubjectData(subject_id, "zhang", ep_rs.astype(np.float32), y, times=times, runs=runs)


def _epoch_continuous(d, fs, epoch_sec=RS_EPOCH_SEC):
    win = int(round(epoch_sec * fs))
    n = d.shape[1] // win
    return d[:, :n * win].reshape(d.shape[0], n, win).transpose(1, 0, 2)


def _load_wang_cond(subject_id, task, tag, max_trials, seed, ordered):
    base = ROOT / "wang" / subject_id / "ses-session1" / "eeg"
    vhdr = base / f"{subject_id}_ses-session1_task-{task}_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    raw = _filt(raw)
    d = raw.get_data() * 1e6
    ep = _epoch_continuous(d, FS_OUT)

    n = ep.shape[0]
    half = n // 2
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(n - half, dtype=np.int64)])

    if not ordered:
        idx = np.random.default_rng(seed).permutation(n)
        ep, y = ep[idx], y[idx]
    if max_trials is not None:
        ep, y = ep[:max_trials], y[:max_trials]
    return SubjectData(subject_id, tag, ep.astype(np.float32), y)


def load_wang_eo_subject(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_wang_cond(subject_id, "eyesopen", "wang_eo", max_trials, seed, ordered)


def load_wang_ec_subject(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_wang_cond(subject_id, "eyesclosed", "wang_ec", max_trials, seed, ordered)


def list_wang_subjects():
    d = ROOT / "wang"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def _load_cogbci_cond(subject_id, cond, tag, max_trials, seed, ordered):
    base = ROOT / "cogbci" / subject_id
    trials, labels = [], []
    for pos, lab in (("Beg", 0), ("End", 1)):
        for set_path in sorted(base.rglob(f"RS_{pos}_*.set")):
            if not set_path.stem.upper().endswith(cond):
                continue
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            raw = _filt(raw)
            d = raw.get_data() * 1e6
            ep = _epoch_continuous(d, FS_OUT)
            trials.append(ep)
            labels.append(np.full(ep.shape[0], lab, dtype=np.int64))

    pooled = np.concatenate(trials, axis=0)
    y = np.concatenate(labels)
    if not ordered:
        idx = np.random.default_rng(seed).permutation(len(y))
        pooled, y = pooled[idx], y[idx]
    if max_trials is not None:
        pooled, y = pooled[:max_trials], y[:max_trials]
    return SubjectData(subject_id, tag, pooled.astype(np.float32), y)


def load_cogbci_eo_subject(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_cogbci_cond(subject_id, "EO", "cogbci_eo", max_trials, seed, ordered)


def load_cogbci_ec_subject(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_cogbci_cond(subject_id, "EC", "cogbci_ec", max_trials, seed, ordered)


def list_cogbci_subjects():
    d = ROOT / "cogbci"
    ids = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    need = set(CH32)
    ok = []
    for sid in ids:
        sets = sorted((d / sid).rglob("RS_*.set"))
        if not sets:
            continue
        complete = True
        for s in sets:
            raw = mne.io.read_raw_eeglab(str(s), preload=False, verbose=False)
            if not need.issubset(raw.ch_names):
                complete = False
                break
        if complete:
            ok.append(sid)
    return ok


def list_cho_subjects():
    return sorted(p.stem for p in (ROOT / "cho").glob("s*.mat"))


def list_lee_subjects(session="session1"):
    files = (ROOT / "lee").glob(f"sess0{session[-1]}_subj*_EEG_MI.mat")
    idx = sorted(int(p.name.split("_subj")[1][:2]) for p in files)
    return [f"sub-{i:02d}" for i in idx]


def list_won_subjects():
    d = ROOT / "won"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("sub-"))


def list_zhang_subjects():
    d = ROOT / "zhang"
    return sorted((p.stem for p in d.glob("S*.mat")), key=lambda s: int(s[1:]))


def _clean_epochs(X, fs, names, onsets_samps, pre_s, task_s, out_fs=FS_OUT):
    from preprocess import clean_continuous
    d, _ = clean_continuous(X, fs, names, out_fs=out_fs, keep=CH32)
    return d


def _robust_keep(task, k):
    peak = np.abs(task).max(axis=(1, 2))
    med = np.median(peak)
    mad = 1.4826 * np.median(np.abs(peak - med))
    return peak <= med + k * mad


def load_cho_subject_clean(subject_id, max_trials=200, peak_uv=3.0):
    path = ROOT / "cho" / f"{subject_id}.mat"
    mat = scipy.io.loadmat(str(path), simplify_cells=True)
    eeg = mat["eeg"]
    n_trials = int(eeg["n_imagery_trials"])
    frame = np.array(eeg["frame"], dtype=np.int32)
    pre = int(abs(int(frame[0])) * CHO_FS // 1000)
    from preprocess import clean_continuous

    trials, bases, labels = [], [], []
    for cls, key in enumerate(("imagery_left", "imagery_right")):
        data = np.array(eeg[key], dtype=np.float64)[:CHO_N_EEG]
        d, _ = clean_continuous(data, CHO_FS, CHO_CH, out_fs=FS_OUT, keep=CH32)
        total = data.shape[1] // n_trials
        total_r = round(total * FS_OUT / CHO_FS)
        pre_r = round(pre * FS_OUT / CHO_FS)
        img_r = round(CHO_IMG_SAMPS * FS_OUT / CHO_FS)
        ep = d[:, :n_trials * total_r].reshape(d.shape[0], n_trials, total_r)
        trials.append(ep[:, :, pre_r:pre_r + img_r])
        bases.append(ep[:, :, :pre_r])
        labels.append(np.full(n_trials, cls, dtype=np.int64))

    task = np.concatenate(trials, axis=1).transpose(1, 0, 2)
    base = np.concatenate(bases, axis=1).transpose(1, 0, 2)
    y = np.concatenate(labels)
    keep = _robust_keep(task, peak_uv)
    task, base, y = task[keep], base[keep], y[keep]
    if max_trials is not None:
        task, base, y = task[:max_trials], base[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "cho", task.astype(np.float32), y, baseline=base.astype(np.float32))


def load_lee_subject_clean(subject_id, session="session1", max_trials=200, peak_uv=3.0):
    from preprocess import clean_continuous
    idx = int(subject_id.split("-")[-1])
    mat = scipy.io.loadmat(str(_lee_path(session, idx)), simplify_cells=True)
    samps = round(LEE_SAMPS * FS_OUT / LEE_FS)
    pre = int(round(2.0 * FS_OUT))
    tasks, bases, ys = [], [], []
    for key in ("EEG_MI_train", "EEG_MI_test"):
        x = np.array(mat[key]["x"], dtype=np.float64).T[:LEE_N_EEG]
        onsets = np.array(mat[key]["t"], dtype=np.int64).ravel()
        y_dec = np.array(mat[key]["y_dec"], dtype=np.int64).ravel()
        d, _ = clean_continuous(x, LEE_FS, LEE_CH, out_fs=FS_OUT, keep=CH32)
        on = np.round(onsets * FS_OUT / LEE_FS).astype(np.int64)
        for o, lab in zip(on, y_dec - 1):
            if o - pre < 0 or o + samps > d.shape[1]:
                continue
            tasks.append(d[:, o:o + samps])
            bases.append(d[:, o - pre:o])
            ys.append(lab)
    task = np.stack(tasks)
    base = np.stack(bases)
    y = np.array(ys, dtype=np.int64)
    keep = _robust_keep(task, peak_uv)
    task, base, y = task[keep], base[keep], y[keep]
    if max_trials is not None:
        task, base, y = task[:max_trials], base[:max_trials], y[:max_trials]
    return SubjectData(subject_id, "lee", task.astype(np.float32), y, baseline=base.astype(np.float32))


def _clean_mod():
    from preprocess import clean_continuous
    return clean_continuous


def load_won_subject_clean(subject_id, max_trials=150, seed=0, k=3.0):
    clean = _clean_mod()
    sub_dir = ROOT / "won" / subject_id
    set_file = sub_dir / "eeg" / f"{subject_id}_task-RSVPtask_run-3_eeg.set"
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    raw.rename_channels({kk: v for kk, v in WON_MAP.items() if kk in raw.ch_names})
    raw.pick_channels([c for c in CH32 if c in raw.ch_names], ordered=True)
    sf = raw.info["sfreq"]
    events, _ = mne.events_from_annotations(raw, verbose=False)
    tgt = events[events[:, 2] == 1]
    ntg = events[events[:, 2] == 2]
    rng = np.random.default_rng(seed)
    n_keep = min(len(ntg), len(tgt) * 2)
    ntg = ntg[rng.choice(len(ntg), size=n_keep, replace=False)]
    all_ev = np.concatenate([tgt, ntg], axis=0)
    all_ev = all_ev[np.argsort(all_ev[:, 0])]
    X = raw.get_data() * 1e6
    d, _ = clean(X, sf, raw.ch_names, out_fs=FS_OUT, keep=None)
    scale = FS_OUT / sf
    pre = int(round(WON_PRE * FS_OUT))
    post = int(round(WON_POST * FS_OUT))
    trials, ys, tms = [], [], []
    for s0, code in zip(all_ev[:, 0], all_ev[:, 2]):
        o = int(round(s0 * scale))
        if o - pre < 0 or o + post > d.shape[1]:
            continue
        w = d[:, o - pre:o + post]
        w = w - w[:, :pre].mean(axis=1, keepdims=True)
        trials.append(w)
        ys.append(1 if code == 1 else 0)
        tms.append(s0 / sf)
    ep = np.stack(trials)
    y = np.array(ys, dtype=np.int64)
    tms = np.array(tms)
    keep = _robust_keep(ep, k)
    ep, y, tms = ep[keep], y[keep], tms[keep]
    if max_trials is not None:
        ep, y, tms = ep[:max_trials], y[:max_trials], tms[:max_trials]
    return SubjectData(subject_id, "won", ep.astype(np.float32), y, times=tms,
                       runs=np.zeros(len(y), dtype=np.int64))


def load_zhang_subject_clean(subject_id, day="Day_1", max_trials=150, seed=0, ordered=False, k=3.0):
    clean = _clean_mod()
    path = ROOT / "zhang" / f"{subject_id}.mat"
    names = [ZHANG_RENAME.get(c, c) for c in ZHANG_CH]
    trials, labels, times, runs = [], [], [], []
    pre_s = int(round(ZHANG_PRE * FS_OUT))
    post_s = int(round(ZHANG_POST * FS_OUT))
    with h5py.File(str(path), "r") as f:
        for run, ref in enumerate(f[day][:, 0]):
            mat = f[ref][:]
            eeg = mat[:, :57].T.astype(np.float64)
            trig = mat[:, 57]
            d, _ = clean(eeg, ZHANG_FS, names, out_fs=FS_OUT, keep=CH32)
            scale = FS_OUT / ZHANG_FS
            for code, lab in ((ZHANG_TARGET, 1), (ZHANG_NONTARGET, 0)):
                for pos in np.where(trig == code)[0]:
                    o = int(round(pos * scale))
                    if o - pre_s < 0 or o + post_s > d.shape[1]:
                        continue
                    trials.append(d[:, o - pre_s:o + post_s])
                    labels.append(lab)
                    times.append(pos)
                    runs.append(run)
    ep = np.stack(trials)
    y = np.array(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tgt_idx = np.where(y == 1)[0]
    ntg_idx = np.where(y == 0)[0]
    n_keep = min(len(ntg_idx), len(tgt_idx) * 2)
    ntg_idx = rng.choice(ntg_idx, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([tgt_idx, ntg_idx]))
    if ordered:
        keep = keep[np.lexsort((np.array(times)[keep], np.array(runs)[keep]))]
    ep, y = ep[keep], y[keep]
    runs = np.array(runs)[keep]
    keep2 = _robust_keep(ep, k)
    times = np.array(times)[keep] / ZHANG_FS
    ep, y, runs, times = ep[keep2], y[keep2], runs[keep2], times[keep2]
    if max_trials is not None:
        ep, y, runs, times = ep[:max_trials], y[:max_trials], runs[:max_trials], times[:max_trials]
    return SubjectData(subject_id, "zhang", ep.astype(np.float32), y, times=times, runs=runs)


def _clean_resting(raw, k=3.0):
    clean = _clean_mod()
    d, _ = clean(raw.get_data() * 1e6, raw.info["sfreq"], raw.ch_names, out_fs=FS_OUT, keep=CH32)
    return d


def _load_wang_cond_clean(subject_id, task, tag, max_trials, seed, ordered, k=3.0):
    base = ROOT / "wang" / subject_id / "ses-session1" / "eeg"
    vhdr = base / f"{subject_id}_ses-session1_task-{task}_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    d = _clean_resting(raw, k)
    ep = _epoch_continuous(d, FS_OUT)
    n = ep.shape[0]
    half = n // 2
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(n - half, dtype=np.int64)])
    keep = _robust_keep(ep, k)
    ep, y = ep[keep], y[keep]
    if not ordered:
        idx = np.random.default_rng(seed).permutation(len(y))
        ep, y = ep[idx], y[idx]
    if max_trials is not None:
        ep, y = ep[:max_trials], y[:max_trials]
    return SubjectData(subject_id, tag, ep.astype(np.float32), y)


def load_wang_eo_subject_clean(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_wang_cond_clean(subject_id, "eyesopen", "wang_eo", max_trials, seed, ordered)


def load_wang_ec_subject_clean(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_wang_cond_clean(subject_id, "eyesclosed", "wang_ec", max_trials, seed, ordered)


def _load_cogbci_cond_clean(subject_id, cond, tag, max_trials, seed, ordered, k=3.0):
    base = ROOT / "cogbci" / subject_id
    trials, labels = [], []
    for pos, lab in (("Beg", 0), ("End", 1)):
        for set_path in sorted(base.rglob(f"RS_{pos}_*.set")):
            if not set_path.stem.upper().endswith(cond):
                continue
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
            d = _clean_resting(raw, k)
            ep = _epoch_continuous(d, FS_OUT)
            trials.append(ep)
            labels.append(np.full(ep.shape[0], lab, dtype=np.int64))
    pooled = np.concatenate(trials, axis=0)
    y = np.concatenate(labels)
    keep = _robust_keep(pooled, k)
    pooled, y = pooled[keep], y[keep]
    if not ordered:
        idx = np.random.default_rng(seed).permutation(len(y))
        pooled, y = pooled[idx], y[idx]
    if max_trials is not None:
        pooled, y = pooled[:max_trials], y[:max_trials]
    return SubjectData(subject_id, tag, pooled.astype(np.float32), y)


def load_cogbci_eo_subject_clean(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_cogbci_cond_clean(subject_id, "EO", "cogbci_eo", max_trials, seed, ordered)


def load_cogbci_ec_subject_clean(subject_id, max_trials=150, seed=0, ordered=False):
    return _load_cogbci_cond_clean(subject_id, "EC", "cogbci_ec", max_trials, seed, ordered)


CLEAN_LOADERS = {
    "cho": load_cho_subject_clean, "lee": load_lee_subject_clean, "won": load_won_subject_clean,
    "zhang": load_zhang_subject_clean, "wang_eo": load_wang_eo_subject_clean,
    "wang_ec": load_wang_ec_subject_clean, "cogbci_eo": load_cogbci_eo_subject_clean,
    "cogbci_ec": load_cogbci_ec_subject_clean,
}
