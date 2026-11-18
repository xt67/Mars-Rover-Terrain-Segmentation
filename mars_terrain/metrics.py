import numpy as np
from .data import CLASSES, IGNORE


def confusion_matrix(prediction, target):
    p, t = np.asarray(prediction), np.asarray(target)
    if p.shape != t.shape:
        raise ValueError('Prediction and target shapes differ.')
    valid = t != IGNORE
    if not np.isin(t[valid], range(4)).all() or not np.isin(p[valid], range(4)).all():
        raise ValueError('Class IDs must be in 0..3.')
    return np.bincount((4 * t[valid].astype(int) + p[valid]).astype(int), minlength=16).reshape(4, 4)


def summarize(cm):
    cm = np.asarray(cm)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.divide(inter, union, out=np.full(4, np.nan), where=union > 0)
    return {'iou': {name: float(v) if np.isfinite(v) else None for name, v in zip(CLASSES, iou)},
            'mean_iou': float(np.nanmean(iou)) if (union > 0).any() else None,
            'pixel_accuracy': float(inter.sum() / cm.sum()) if cm.sum() else None,
            'labeled_pixels': int(cm.sum()), 'confusion_matrix': cm.tolist(),
            'absent_class_policy': 'Exclude zero-union classes from mean IoU; report null.'}
