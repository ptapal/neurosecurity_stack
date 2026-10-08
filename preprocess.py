import mne
import numpy as np

mne.set_log_level("ERROR")

FRONTAL = ("Fp1", "Fp2")
MAX_EXCLUDE = 6


def clean_continuous(X, fs, ch_names, out_fs=250.0, keep=None):  # X, return: channels x samples, uV
    info = mne.create_info(ch_names=list(ch_names), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(X * 1e-6, info, verbose=False)
    if keep is not None:
        raw.pick_channels([c for c in keep if c in raw.ch_names], ordered=True)
    raw.filter(1.0, 40.0, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    if out_fs is not None and out_fs != fs:
        raw.resample(out_fs, verbose=False)

    data = raw.get_data() * 1e6
    std = data.std(axis=1)
    med = np.median(std)
    bads = [raw.ch_names[i] for i in range(len(std)) if std[i] > 4 * med or std[i] < 0.2 * med]
    raw.info["bads"] = bads
    if bads:
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore",
                        match_case=False, verbose=False)
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception:
            raw.info["bads"] = []

    raw.set_eeg_reference("average", projection=False, verbose=False)

    n_comp = min(20, len(raw.ch_names) - 1)
    ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=0,
                                max_iter=500, verbose=False)
    ica.fit(raw, verbose=False)
    frontal = [c for c in FRONTAL if c in raw.ch_names] or [raw.ch_names[0]]
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=frontal, threshold=3.0, verbose=False)
    try:
        muscle_idx, _ = ica.find_bads_muscle(raw, threshold=0.6, verbose=False)
    except Exception:
        muscle_idx = []
    exclude = sorted(set(eog_idx) | set(muscle_idx))[:MAX_EXCLUDE]
    ica.exclude = exclude
    ica.apply(raw, verbose=False)

    return raw.get_data() * 1e6, {"bads": bads, "ica_excluded": exclude, "n_components": n_comp}
