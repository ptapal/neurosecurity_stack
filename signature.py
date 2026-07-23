import struct
from dataclasses import dataclass

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def make_projection_matrix(r, s, rng):
    return rng.normal(size=(s, r)) / np.sqrt(s)


def dp_project(z, P, sigma_s, rng):
    proj = z @ P.T
    frob = float(np.linalg.norm(P, "fro"))
    return proj + rng.normal(0.0, sigma_s * frob, size=proj.shape)


@dataclass
class LSHParams:
    a: np.ndarray
    bias: np.ndarray
    width: float


def make_lsh(s, b, rng):
    a = rng.normal(size=(b, s))
    width = 2.0 * np.sqrt(s)
    bias = rng.uniform(0, width, size=b)
    return LSHParams(a, bias, width)


def bucket_ids(z, p):
    return np.floor((z @ p.a.T + p.bias) / p.width).astype(np.int64)


def enroll_reference(z_ref, p):
    return bucket_ids(z_ref[None, :], p)[0]


def fingerprint(z, p, ref_ids):
    ids = bucket_ids(z, p)
    return (ids == ref_ids[None, :]).astype(np.uint8)


def mismatch_frac(bits):
    return 1.0 - np.mean(bits, axis=-1)


def calibrate_threshold(mismatch_clean, target_frr):
    return float(np.quantile(mismatch_clean, 1.0 - target_frr))


def verify_tamper(mismatch, tau):
    return mismatch <= tau


@dataclass
class DeviceKey:
    client_id: str
    sk: Ed25519PrivateKey
    pk: Ed25519PublicKey


def keygen(client_id):
    sk = Ed25519PrivateKey.generate()
    return DeviceKey(client_id, sk, sk.public_key())


def _payload(bits, ts, client_id):
    return np.packbits(bits).tobytes() + struct.pack(">Q", ts) + client_id.encode()


def sign_reading(dev, bits, ts):
    return dev.sk.sign(_payload(bits, ts, dev.client_id))


def verify_signature(dev, bits, ts, sig):
    try:
        dev.pk.verify(sig, _payload(bits, ts, dev.client_id))
        return True
    except InvalidSignature:
        return False
