"""Launch: python -m streamlit run app.py"""
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from cv_common.fixtures import terrain_fixture

st.set_page_config(page_title='Mars-Rover-Terrain-Segmentation', page_icon='🪐', layout='wide')
st.markdown('''<style>
.stApp {background:#080e17;color:#e5edf7} [data-testid="stSidebar"] {background:#101927}
.block-container {max-width:1360px;padding-top:2.2rem} h1 {letter-spacing:-.045em;font-weight:750!important}
[data-testid="stMetric"] {border:1px solid #28364a;border-radius:14px;padding:16px;background:#111c2b}
.eyebrow {font:12px monospace;letter-spacing:.2em;color:#efad76;margin-bottom:12px}
.intro {color:#9aacc1;font-size:17px;max-width:780px}.rule {border-top:1px solid #28364a;margin:24px 0}
div.stButton>button[kind="primary"] {background:#e4965e;color:#10151d;border:0}
</style>''', unsafe_allow_html=True)
st.sidebar.markdown('### 🪐 ORBITAL / VISION')
st.sidebar.caption('SPACE & COMPUTER VISION PROJECT')
project = 'Mars terrain'
st.sidebar.markdown('Mars terrain')
st.sidebar.markdown('---')
st.sidebar.caption('Rayan Rahman · Research workbench')
st.sidebar.caption('Synthetic samples verify the pipeline. They are not trained model predictions or real mission results.')
st.markdown('<div class="eyebrow">ORBITAL VISION LAB / EXPERIMENT WORKSPACE</div>', unsafe_allow_html=True)


def image_bytes(im):
    buf = io.BytesIO(); im.save(buf, format='PNG'); return buf.getvalue()


def safe_image(upload):
    with Image.open(upload) as im:
        if im.width * im.height > 24_000_000: raise ValueError('Use an image under 24 megapixels.')
        return im.convert('RGB')


def error_box(exc):
    st.error(str(exc))
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        st.code('python -m pip install -r requirements.txt')


from mars_terrain.data import CLASSES, PALETTE
from mars_terrain.predict import overlay, predict
st.title('Read the Martian surface.')
st.markdown('<p class="intro">Separate soil, bedrock, sand and big rock. Inspect class coverage, export pixel labels, and evaluate the terrain that matters.</p>', unsafe_allow_html=True)
inspect, train, results = st.tabs(['Terrain explorer', 'Train a model', 'Evaluation'])
with inspect:
    left, right = st.columns([1, 2.1], gap='large')
    with left:
        source = st.radio('Input', ['Synthetic mask example', 'Upload rover image'])
        alpha = st.slider('Overlay opacity', 0., 1., .5)
        for name, color in zip(CLASSES, PALETTE):
            st.markdown(f'<span style="color:rgb{tuple(int(c) for c in color)}">●</span> {name.title()}', unsafe_allow_html=True)
        if source == 'Synthetic mask example':
            image, labels = terrain_fixture(); st.caption('Known synthetic labels. No model inference. Dark top strip is unlabeled (255).')
        else:
            checkpoint = st.text_input('Trained checkpoint', 'runs/mars/best.pt')
            upload = st.file_uploader('Rover image', type=['png', 'jpg', 'jpeg'])
            image = labels = None
            if upload:
                try:
                    image = safe_image(upload)
                    if st.button('Segment terrain', type='primary'):
                        from mars_terrain.model import load_checkpoint
                        with st.spinner('Running trained terrain model…'):
                            model, meta = load_checkpoint(checkpoint)
                            labels, confidence = predict(model, image, meta['size'])
                            st.session_state['mars_result'] = (upload.getvalue(), checkpoint, labels, confidence)
                    cached = st.session_state.get('mars_result')
                    if cached and cached[0] == upload.getvalue() and cached[1] == checkpoint: labels = cached[2]
                except Exception as exc: error_box(exc)
    with right:
        if image is not None:
            if labels is not None:
                col1, col2 = st.columns(2)
                col1.image(image, caption='Input image', width='stretch')
                col2.image(overlay(image, labels, alpha), caption='Terrain overlay', width='stretch')
                valid = labels < 4; counts = [int((labels == i).sum()) for i in range(4)]
                cols = st.columns(4)
                for col, name, count in zip(cols, CLASSES, counts): col.metric(name.title(), f'{count / max(1, valid.sum()):.1%}')
                st.caption('Coverage is over labeled/predicted pixels; it is not accuracy or a safety score.')
                st.download_button('Download overlay', image_bytes(overlay(image, labels, alpha)), 'terrain_overlay.png', 'image/png')
                st.download_button('Download raw class mask', image_bytes(Image.fromarray(labels)), 'terrain_labels.png', 'image/png')
            else: st.image(image, caption='Ready for a trained checkpoint', width='stretch')
        else: st.info('Upload an image and provide a trained checkpoint to begin.')
with train:
    st.subheader('Train on AI4Mars')
    st.write('DeepLabV3+ with a ResNet-50 encoder. Training excludes rover, distant and unlabeled pixels. Validation is drawn only from the training split.')
    st.code('python -m mars_terrain.train --epochs 20 --batch-size 4 --size 256')
    st.write('The first run downloads the dataset and ImageNet encoder weights. Use a GPU. The notebook in notebooks/ contains the same workflow for Colab.')
    st.code('python -m mars_terrain.gradio_app --checkpoint runs/mars/best.pt')
with results:
    metrics_file = Path('outputs/mars-evaluation/metrics.json')
    if metrics_file.is_file(): st.json(json.loads(metrics_file.read_text(encoding='utf-8')))
    else: st.info('No expert evaluation has been run. No accuracy claims are available yet.')
    st.code('python -m mars_terrain.evaluate --checkpoint runs/mars/best.pt')
    st.caption('Reports class IoU, mean IoU, pixel accuracy, confusion matrices and all-soil baselines on test_min1 and test_min3. Saves the worst sand misses for inspection.')
