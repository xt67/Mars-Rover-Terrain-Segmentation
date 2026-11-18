"""Training/evaluation lifecycle with a tiny test model and synthetic local masks."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from cv_common.fixtures import terrain_fixture


class TrainingTests(unittest.TestCase):
    def test_training_checkpoint_and_expert_evaluation(self):
        import torch
        from mars_terrain.train import main as train
        from mars_terrain.evaluate import main as evaluate
        def tiny(*args, **kwargs):
            return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 4, 1))
        with tempfile.TemporaryDirectory(prefix='space-cv-test-') as temp:
            root = Path(temp); image, label = terrain_fixture(64)
            image.save(root / 'image.png'); Image.fromarray(label).save(root / 'label.png')
            entries = [{'split': 'train', 'image': 'image.png', 'label_mask': 'label.png'} for _ in range(6)]
            entries += [{'split': s, 'image': 'image.png', 'label_mask': 'label.png'} for s in ['test_min1', 'test_min3']]
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps(entries), encoding='utf-8')
            with patch('mars_terrain.train.build_model', side_effect=tiny), patch('sys.stdout', new=io.StringIO()):
                train(['--manifest', str(manifest), '--output', str(root / 'run'), '--epochs', '1', '--size', '64', '--batch-size', '2', '--device', 'cpu', '--no-pretrained', '--backend', 'native'])
            self.assertTrue((root / 'run/best.pt').is_file())
            config = json.loads((root / 'run/config.json').read_text())
            self.assertFalse(set(config['train_indices']) & set(config['validation_indices']))
            with patch('mars_terrain.model.build_model', side_effect=tiny), patch('sys.stdout', new=io.StringIO()):
                result = evaluate(['--manifest', str(manifest), '--checkpoint', str(root / 'run/best.pt'), '--output', str(root / 'eval')])
            self.assertGreater(result['test_min3']['model']['labeled_pixels'], 0)
            self.assertTrue((root / 'eval/metrics.json').is_file())


if __name__ == '__main__': unittest.main()
