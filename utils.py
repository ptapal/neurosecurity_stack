import json
import zlib

from config import RESULTS_DIR


def seed_from(*parts):
    return zlib.crc32(repr(parts).encode()) % (2 ** 32)  # deterministic, unlike Python's salted hash()


def load_json(name):
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def save_json(name, obj):
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
