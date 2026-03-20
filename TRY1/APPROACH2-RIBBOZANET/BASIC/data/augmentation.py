"""
augmentation.py — Random rotation and translation of 3D structures.

Physics doesn't depend on the viewing angle. Randomly rotating and
translating training structures teaches the model this invariance
and helps prevent overfitting.
"""

import numpy as np


def random_rotation(coords: np.ndarray) -> np.ndarray:
    """Apply a uniformly random 3D rotation to coordinates.

    Uses QR decomposition of a random matrix to generate a
    uniformly distributed rotation matrix (Haar measure).

    Args:
        coords: (N, 3) array of 3D coordinates.

    Returns:
        rotated: (N, 3) rotated coordinates.
    """
    # Generate random 3x3 matrix from standard normal
    M = np.random.randn(3, 3)

    # QR decomposition gives an orthogonal matrix Q
    Q, R = np.linalg.qr(M)

    # Ensure proper rotation (det = +1, not -1 which would be a reflection)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    # Apply rotation: each point (row) is multiplied by Q^T
    rotated = coords @ Q.T

    return rotated


def random_translation(coords: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """Apply a random translation to coordinates.

    Args:
        coords: (N, 3) array of 3D coordinates.
        scale: Standard deviation of the random shift (in Angstroms).

    Returns:
        translated: (N, 3) shifted coordinates.
    """
    shift = np.random.randn(3) * scale
    return coords + shift
