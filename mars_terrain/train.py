import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
from .data import CLASSES, IGNORE, TerrainDataset, prepare, load_rows, split_indices, class_weights, tensor_array, as_tensor
from .metrics import confusion_matrix, summarize
from .model import build_model


def main(argv=None):
    p = argparse.ArgumentParser(description='Train DeepLabV3+ without touching expert test splits.')
    p.add_argument('--manifest', help='Optional local JSON manifest; otherwise downloads AI4MARS.')
    p.add_argument('--output', default='runs/mars')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backend', choices=['smp', 'native'], default='smp')
    p.add_argument('--no-pretrained', action='store_true', help='Random initialization, no encoder download.')
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, help='Optional labeled training subset; metrics must be described as subset results.')
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.size < 64 or args.size % 32 or args.lr <= 0:
        p.error('epochs >= 1, batch-size >= 2, size >= 64 divisible by 32, lr > 0 required.')
    import torch
    from torch.utils.data import DataLoader
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    rows = load_rows('train', args.manifest)
    eligible = [i for i in range(len(rows)) if np.any(prepare(rows[i])[1] != IGNORE)]
    if args.limit:
        eligible = np.random.default_rng(args.seed).permutation(eligible)[:args.limit].tolist()
    tr, va = split_indices(len(eligible), seed=args.seed)
    train_ids, val_ids = [eligible[i] for i in tr], [eligible[i] for i in va]
    if len(train_ids) < 2:
        raise ValueError('Need at least two training images after the validation split.')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'history.csv').exists():
        raise FileExistsError('Output already contains a run. Choose a new --output directory.')
    counts = np.zeros(4, np.int64)
    for i in train_ids:
        _, label = prepare(rows[i])
        counts += np.bincount(label[label != IGNORE], minlength=4)
    weights = class_weights(counts)
    config = vars(args) | {'train_indices': train_ids, 'validation_indices': val_ids, 'class_counts': counts.tolist(),
                           'class_weights': weights.tolist(), 'device_used': device}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train = DataLoader(TerrainDataset(rows, train_ids, args.size, True), batch_size=min(args.batch_size, len(train_ids)), shuffle=True, drop_last=True)
    val = DataLoader(TerrainDataset(rows, val_ids, args.size), batch_size=args.batch_size)
    model = build_model(args.backend, not args.no_pretrained).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=as_tensor(weights).to(device), ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.; steps = 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            if not (y != IGNORE).any(): continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss.')
            loss.backward(); opt.step()
            total_loss += loss.item(); steps += 1
        if steps == 0: raise ValueError('No valid training batches.')
        model.eval(); cm = np.zeros((4, 4), np.int64)
        with torch.inference_mode():
            for x, y in val:
                pred = tensor_array(model(x.to(device)).argmax(1))
                cm += confusion_matrix(pred, tensor_array(y))
        score = summarize(cm)
        if score['mean_iou'] is None: raise ValueError('Validation split has no valid pixels after resizing.')
        record = {'epoch': epoch, 'loss': total_loss / steps, 'val_mean_iou': score['mean_iou'],
                  'val_sand_iou': score['iou']['sand'], 'val_pixel_accuracy': score['pixel_accuracy']}
        with (out / 'history.csv').open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if epoch == 1: writer.writeheader()
            writer.writerow(record)
        if score['mean_iou'] > best:
            best = score['mean_iou']
            torch.save({'kind': 'ai4mars-trained', 'backend': args.backend, 'classes': CLASSES,
                        'state_dict': model.state_dict(), 'size': args.size, 'epoch': epoch,
                        'validation': score, 'seed': args.seed, 'manifest': args.manifest,
                        'training_subset_limit': args.limit}, out / 'best.pt')
        print(json.dumps(record), flush=True)
    return out


if __name__ == '__main__': main()
