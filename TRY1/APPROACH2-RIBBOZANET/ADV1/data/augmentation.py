"""
augmentation.py — Random rotation and translation of 3D structures.
IDENTICAL to BASIC/data/augmentation.py — no changes needed for ADV1.
"""

import numpy as np


def random_rotation(coords: np.ndarray) -> np.ndarray:
    M = np.random.randn(3, 3)
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return coords @ Q.T


def random_translation(coords: np.ndarray, scale: float = 10.0) -> np.ndarray:
    shift = np.random.randn(3) * scale
    return coords + shift
