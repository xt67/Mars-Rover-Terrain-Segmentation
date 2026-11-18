"""One synthetic optimization step: architecture wiring check, not training evidence."""
import json
from pathlib import Path
import torch
from mars_terrain.model import build_model
from mars_terrain.data import TerrainDataset
from cv_common.fixtures import terrain_fixture


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model = build_model('native', pretrained=False).to(device).train()
    image, label = terrain_fixture(64)
    dataset = TerrainDataset([{'image': image, 'label_mask': label}] * 2, size=64)
    x = torch.stack([dataset[i][0] for i in range(2)]).to(device)
    y = torch.stack([dataset[i][1] for i in range(2)]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pred = model(x); loss = torch.nn.functional.cross_entropy(pred, y, ignore_index=255)
    loss.backward(); optimizer.step()
    assert tuple(pred.shape) == (2, 4, 64, 64)
    assert torch.isfinite(loss)
    report = {'test': 'Native DeepLabV3+ synthetic forward/backward/optimizer step', 'device': device,
              'output_shape': list(pred.shape), 'loss': loss.item(), 'real_ai4mars_training': False}
    Path('verification').mkdir(exist_ok=True)
    Path('verification/model_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
