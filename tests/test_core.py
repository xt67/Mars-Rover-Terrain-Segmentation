import unittest
import numpy as np
from PIL import Image
from cv_common.fixtures import terrain_fixture
from mars_terrain.data import prepare, split_indices, class_weights, resize_pair
from mars_terrain.metrics import confusion_matrix, summarize

class TerrainTests(unittest.TestCase):

    def test_ignored_regions_and_rgb_masks(self):
        image = Image.new('RGB', (3, 2))
        label = np.array([[0, 1, 2], [3, 255, 0]], np.uint8)
        _, actual = prepare({'image': image, 'label_mask': np.repeat(label[..., None], 3, -1), 'rover_mask': np.array([[0, 1, 0], [0, 0, 0]]), 'range_mask': np.array([[0, 0, 0], [1, 0, 0]])})
        np.testing.assert_array_equal(actual, [[0, 255, 2], [255, 255, 0]])

    def test_ignore_not_scored(self):
        cm = confusion_matrix(np.array([0, 0, 1, 3, 99]), np.array([0, 1, 1, 3, 255]))
        score = summarize(cm)
        self.assertEqual(score['pixel_accuracy'], 0.75)
        self.assertEqual(score['iou']['soil'], 0.5)
        self.assertIsNone(score['iou']['sand'])
        self.assertAlmostEqual(score['mean_iou'], 2 / 3)

    def test_all_ignore(self):
        self.assertIsNone(summarize(confusion_matrix(np.zeros(3, int), np.full(3, 255)))['mean_iou'])

    def test_split_is_disjoint_reproducible(self):
        tr, va = split_indices(100)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(len(va), 10)
        self.assertEqual((tr, va), split_indices(100))

    def test_masks_resize_nearest(self):
        image, label = terrain_fixture(64)
        _, small = resize_pair(np.asarray(image), label, 31)
        self.assertTrue(set(np.unique(small)) <= {0, 1, 2, 3, 255})

    def test_class_weights_and_bad_labels(self):
        np.testing.assert_allclose(class_weights([70, 10, 10, 10]), [0.3, 0.9, 0.9, 0.9])
        with self.assertRaises(ValueError):
            class_weights([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            prepare({'image': Image.new('RGB', (2, 2)), 'label_mask': np.full((2, 2), 5)})
