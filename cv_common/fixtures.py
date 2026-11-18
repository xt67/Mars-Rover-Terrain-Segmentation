"""Deterministic synthetic fixtures, never represented as real mission data."""
import cv2
import numpy as np
from PIL import Image

def terrain_fixture(size=256):
    y, x = np.mgrid[:size, :size]
    label = np.zeros((size, size), np.uint8)
    label[(y > size * 0.42) & (x < size * 0.43)] = 1
    label[(x > size * 0.55) & (y > size * 0.25)] = 2
    label[(x - size * 0.42) ** 2 + (y - size * 0.68) ** 2 < (size * 0.13) ** 2] = 3
    rng = np.random.default_rng(8)
    shades = np.array([[163, 110, 74], [115, 99, 88], [194, 152, 100], [84, 73, 65]])
    image = np.clip(shades[label] + rng.normal(0, 9, (size, size, 1)), 0, 255).astype(np.uint8)
    label[:size // 7] = 255
    return (Image.fromarray(image), label)
