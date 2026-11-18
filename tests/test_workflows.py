import io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from cv_common.fixtures import terrain_fixture
from mars_terrain.evaluate import main as evaluate

class WorkflowTests(unittest.TestCase):

    def test_local_expert_baseline(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='space-cv-test-') as temp:
            root = Path(temp)
            image, label = terrain_fixture(64)
            image.save(root / 'image.png')
            Image.fromarray(label).save(root / 'label.png')
            manifest = [{'split': split, 'image': 'image.png', 'label_mask': 'label.png'} for split in ['test_min1', 'test_min3']]
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch('sys.stdout', new=io.StringIO()):
                result = evaluate(['--manifest', str(root / 'manifest.json'), '--baseline-only', '--output', str(root / 'evaluation')])
            self.assertEqual(set(result), {'test_min1', 'test_min3'})
            self.assertEqual(result['test_min3']['all_soil']['iou']['sand'], 0.0)
            self.assertEqual(result['test_min3']['all_soil']['labeled_pixels'], int((label != 255).sum()))
