"""Standalone Gradio upload demo requested by the project guide."""
import argparse
from .model import load_checkpoint
from .predict import predict, overlay
from .data import CLASSES
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', default='runs/mars/best.pt'); a = p.parse_args()
    model, meta = load_checkpoint(a.checkpoint)
    import gradio as gr
    def infer(image, alpha):
        if image is None: raise gr.Error('Upload a rover image.')
        labels, confidence = predict(model, image, meta['size'])
        return overlay(image, labels, alpha), {name: float(np.mean(labels == i)) for i, name in enumerate(CLASSES)}
    gr.Interface(infer, [gr.Image(type='pil'), gr.Slider(0, 1, value=.5, label='Overlay opacity')],
                 [gr.Image(label='Terrain overlay'), gr.JSON(label='Predicted pixel fractions')],
                 title='Mars Rover Terrain Segmentation',
                 description='DeepLabV3+ terrain labels. Sand is yellow. Research demo; predictions do not establish safe driving routes.').launch(server_name='127.0.0.1')


if __name__ == '__main__': main()
