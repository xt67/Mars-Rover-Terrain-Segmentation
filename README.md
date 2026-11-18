# Mars Rover Terrain Segmentation

Label rover images as soil, bedrock, sand and big rock using DeepLabV3+ with a ResNet-50 encoder.

Built for **Rayan Rahman** as part of the Space And Computer Vision Project.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The app includes a labeled synthetic example; no downloads are needed to explore that example after installing the base dependencies. On macOS/Linux, use `.venv/bin/python`.

## Workflow


Implements Project 2, pages 5–7, of the supplied `Space_ML_Project_Guides.pdf`.

```powershell
python -m mars_terrain.train --epochs 20 --batch-size 4 --size 256
python -m mars_terrain.evaluate --checkpoint runs/mars/best.pt
python -m mars_terrain.predict input/rover.jpg --checkpoint runs/mars/best.pt
python -m mars_terrain.gradio_app --checkpoint runs/mars/best.pt
```

The default backend is `segmentation-models-pytorch`, as specified by the guide. `--backend native` uses a separately implemented DeepLabV3+ decoder with torchvision's ResNet-50 and ASPP. Checkpoints record the backend, and evaluation restores that exact implementation. `--no-pretrained` disables the ImageNet encoder download; it does not make a random model suitable for inference. Batch size must be at least 2 for batch normalization.

Features:

- Soil 0, bedrock 1, sand 2, big rock 3; ignore 255.
- Rover-body and >30m masks are applied before training or scoring.
- Reproducible 90/10 training/validation split; expert test splits never used for checkpoint selection.
- Counts and class weights `1 - class_fraction` computed on the training portion only.
- ImageNet normalization, paired horizontal flips, image-only brightness augmentation; nearest-neighbor mask resize.
- Weighted cross entropy with ignored pixels; all-ignore batches skipped.
- Best validation-mIoU checkpoint, CSV training history, split indices and configuration recorded.
- Evaluation at original image/mask resolution on `test_min1` and `test_min3`; all-soil baseline, class IoU, mIoU, accuracy, confusion matrix and worst sand misses.
- Zero-union classes report `null` and are excluded from mIoU; a class with false-positive predictions has nonzero union and is scored.

The Hugging Face dataset is approximately 6.5 GB, excluding cache expansion. Training may take hours; use a GPU. `--limit 100` is useful for integration checks, but subset results must be labeled. Nearby views may still be correlated in an image-level validation split; the expert evaluation remains the final benchmark.

### Local dataset option

Pass `--manifest data/manifest.json` to training and evaluation. Paths are relative to the manifest file. Each row must specify a split and an image; training labels are required for usable rows. Optional masks mark excluded regions with positive values.

```json
[
  {"split":"train", "image":"images/001.jpg", "label_mask":"labels/001.png", "rover_mask":"rover/001.png", "range_mask":"range/001.png"},
  {"split":"test_min1", "image":"images/test.jpg", "label_mask":"expert_min1/test.png"},
  {"split":"test_min3", "image":"images/test.jpg", "label_mask":"expert_min3/test.png"}
]
```

Use enough labeled training rows for both train and validation. The manifest above illustrates the schema, not a sufficient dataset. Grayscale IDs or replicated RGB IDs are accepted, not palette visualization colors. Maintain the official expert splits. The pipeline records file provenance but cannot establish whether arbitrary user-supplied labels are genuine AI4Mars labels.

`notebooks/Mars_Training_Colab.ipynb` provides the cloud-GPU workflow. No trained checkpoint is bundled.


## Verification and limitations

Full AI4Mars training and expert-set evaluation have not been run. No trained checkpoint or claimed real-data accuracy is bundled. The native model passed a CUDA forward/backward step; the default SMP backend still needs its dependencies and dataset downloads.

Synthetic examples are pipeline checks, not real-world benchmark results. See `verification/STATUS.md`.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m tools.verify_ui
```

## Sources

Based on the supplied Space_ML_Project_Guides.pdf, Mars project pages 5–7 (source-guide credit: @byswergiu1).

- https://huggingface.co/datasets/hassanjbara/AI4MARS
- https://smp.readthedocs.io/en/latest/models.html
