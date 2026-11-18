import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .data import load_rows, prepare
from .metrics import confusion_matrix, summarize
from .model import load_checkpoint
from .predict import predict, overlay


def main(argv=None):
    p = argparse.ArgumentParser(description='Final expert evaluation at original mask resolution.')
    p.add_argument('--checkpoint', default='runs/mars/best.pt')
    p.add_argument('--manifest'); p.add_argument('--output', default='outputs/mars-evaluation')
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--splits', nargs='+', choices=['test_min1', 'test_min3'], default=['test_min1', 'test_min3'])
    a = p.parse_args(argv)
    model, meta = (None, {}) if a.baseline_only else load_checkpoint(a.checkpoint)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in a.splits:
        rows = load_rows(split, a.manifest)
        if not len(rows): raise ValueError(f'No rows for {split}.')
        base = np.zeros((4, 4), np.int64); cm = base.copy(); errors = []
        for i in range(len(rows)):
            image, target = prepare(rows[i])
            base += confusion_matrix(np.zeros_like(target), target)
            if model is not None:
                pred, _ = predict(model, Image.fromarray(image), meta['size'])
                cm += confusion_matrix(pred, target)
                sand = target == 2
                if sand.any(): errors.append({'row': i, 'sand_miss_rate': float((pred[sand] != 2).mean()), 'sand_pixels': int(sand.sum())})
        results[split] = {'images': len(rows), 'all_soil': summarize(base)}
        if model is not None:
            results[split]['model'] = summarize(cm)
            errors.sort(key=lambda x: x['sand_miss_rate'], reverse=True)
            results[split]['worst_sand_errors'] = errors[:10]
            for item in errors[:5]:
                image, label = prepare(rows[item['row']])
                im = Image.fromarray(image); pred, _ = predict(model, im, meta['size'])
                overlay(im, pred).save(out / f'{split}-sand-error-{item["row"]}.png')
                overlay(im, label).save(out / f'{split}-truth-{item["row"]}.png')
    (out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2)); return results


if __name__ == '__main__': main()
