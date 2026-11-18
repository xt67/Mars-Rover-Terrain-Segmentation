import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .data import PALETTE, image_tensor, tensor_array
from .model import load_checkpoint


def predict(model, image, size=256, device='cpu'):
    import torch
    from torch.nn import functional as F
    image = image.convert('RGB')
    x = image_tensor(image.resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = F.interpolate(model(x), size=(image.height, image.width), mode='bilinear', align_corners=False)
        probabilities = torch.softmax(logits, 1)
        confidence, labels = probabilities.max(1)
    return tensor_array(labels[0]).astype(np.uint8), tensor_array(confidence[0])


def overlay(image, labels, alpha=.5):
    rgb = np.asarray(image.convert('RGB'))
    valid = labels < 4
    result = rgb.copy()
    result[valid] = ((1 - alpha) * rgb[valid] + alpha * PALETTE[labels[valid]]).astype(np.uint8)
    return Image.fromarray(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image'); p.add_argument('--checkpoint', default='runs/mars/best.pt'); p.add_argument('--output', default='outputs/mars')
    a = p.parse_args()
    model, meta = load_checkpoint(a.checkpoint)
    image = Image.open(a.image).convert('RGB')
    labels, confidence = predict(model, image, meta['size'])
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(out / 'labels.png')
    overlay(image, labels).save(out / 'overlay.png')
    np.save(out / 'confidence.npy', confidence)


if __name__ == '__main__': main()
