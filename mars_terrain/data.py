from pathlib import Path
import json
import random
import numpy as np
from PIL import Image, ImageEnhance

CLASSES = ['soil', 'bedrock', 'sand', 'big rock']
IGNORE = 255
PALETTE = np.array([[184, 116, 77], [105, 145, 175], [247, 199, 76], [204, 89, 119]], np.uint8)


def mask_array(value):
    arr = np.asarray(value)
    if arr.ndim == 3:
        if not np.all(arr[..., :3] == arr[..., :1]):
            raise ValueError('Mask must contain replicated class IDs, not arbitrary RGB colors.')
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError('Mask must be two dimensional.')
    return arr


def prepare(row):
    image = np.asarray(row['image'].convert('RGB'))
    if row.get('label_mask') is None:
        label = np.full(image.shape[:2], IGNORE, np.uint8)
    else:
        raw = mask_array(row['label_mask'])
        if not np.isin(raw, [0, 1, 2, 3, IGNORE]).all():
            raise ValueError('Unexpected label IDs; expected 0..3 and 255.')
        label = raw.astype(np.uint8).copy()
    if label.shape != image.shape[:2]:
        raise ValueError('Image and label dimensions differ.')
    for key in ('rover_mask', 'range_mask'):
        if row.get(key) is not None:
            m = mask_array(row[key])
            if m.shape != label.shape:
                raise ValueError(f'{key} dimensions differ from image.')
            label[m > 0] = IGNORE
    return image, label


def split_indices(n, fraction=.1, seed=42):
    if n < 2 or not 0 < fraction < 1:
        raise ValueError('Need at least two labeled training images and 0 < validation fraction < 1.')
    ids = np.random.default_rng(seed).permutation(n)
    cut = max(1, min(n - 1, round(n * fraction)))
    return ids[cut:].tolist(), ids[:cut].tolist()


def class_weights(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (4,) or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError('Training split has no valid labeled pixels.')
    # Clamp only the degenerate one-class case to avoid zero total loss weight.
    return np.maximum(1 - counts / counts.sum(), 1e-6).astype(np.float32)


def resize_pair(image, label, size):
    return (np.array(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)),
            np.array(Image.fromarray(label).resize((size, size), Image.Resampling.NEAREST)))


def as_tensor(arr):
    import torch
    arr = np.ascontiguousarray(arr)
    try:
        return torch.from_numpy(arr)
    except RuntimeError:
        # Older torch wheels cannot bridge NumPy 2; byte buffer avoids that ABI.
        dtype = torch.float32 if arr.dtype == np.float32 else torch.int64
        return torch.frombuffer(bytearray(arr.tobytes()), dtype=dtype).reshape(arr.shape).clone()


def tensor_array(tensor):
    t = tensor.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist())


def image_tensor(image):
    a = np.asarray(image, dtype=np.float32) / 255
    a = (a - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return as_tensor(a.transpose(2, 0, 1).copy())


class TerrainDataset:
    def __init__(self, rows, indices=None, size=256, augment=False):
        self.rows = rows
        self.indices = list(range(len(rows))) if indices is None else indices
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = prepare(self.rows[self.indices[i]])
        image, label = resize_pair(image, label, self.size)
        if self.augment:
            if random.random() < .5:
                image, label = image[:, ::-1].copy(), label[:, ::-1].copy()
            image = np.array(ImageEnhance.Brightness(Image.fromarray(image)).enhance(random.uniform(.8, 1.2)))
        return image_tensor(image), as_tensor(label.astype(np.int64))


def load_rows(split, manifest=None):
    if manifest:
        path = Path(manifest).resolve()
        entries = json.loads(path.read_text(encoding='utf-8'))
        selected = [e for e in entries if e['split'] == split]
        class LocalRows:
            def __len__(self): return len(selected)
            def __getitem__(self, i):
                row = dict(selected[i])
                for key in ('image', 'label_mask', 'rover_mask', 'range_mask'):
                    if row.get(key):
                        with Image.open(path.parent / row[key]) as im:
                            row[key] = im.copy()
                return row
        return LocalRows()
    from datasets import load_dataset
    ds = load_dataset('hassanjbara/AI4MARS', split=split)
    if split == 'train':
        ds = ds.filter(lambda row: bool(row['has_labels']))
    return ds
