"""Shared paired geometry for training image, water, and valid arrays."""

import numpy as np


def augment_triplet(image: np.ndarray, water: np.ndarray,
                    valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same rotation and flips to three channel-first arrays."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k=k, axes=(1, 2))
    water = np.rot90(water, k=k, axes=(1, 2))
    valid = np.rot90(valid, k=k, axes=(1, 2))
    if np.random.random() < .5:
        image = np.flip(image, axis=2)
        water = np.flip(water, axis=2)
        valid = np.flip(valid, axis=2)
    if np.random.random() < .5:
        image = np.flip(image, axis=1)
        water = np.flip(water, axis=1)
        valid = np.flip(valid, axis=1)
    return image, water, valid
