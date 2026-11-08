"""Run against the actual Week 6 checkpoint and X-ray: pytest -q test_api.py."""
import io
import os

import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

import app as api

SAMPLE = os.getenv('SAMPLE_IMAGE', 'sample.jpeg')


def post_image(client):
    with open(SAMPLE, 'rb') as image:
        return client.post('/predict', files={'file': ('sample.jpeg', image, 'image/jpeg')})


def test_health():
    with TestClient(api.app) as c:
        response = c.get('/health')
        assert response.status_code == 200
        assert response.json()['model_loaded']
        assert api.POSITIVE in response.json()['classes']


def test_predict_schema_and_probabilities():
    with TestClient(api.app) as c:
        response = post_image(c)
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {'label', 'probabilities', 'latency_ms'}
        assert body['label'] in body['probabilities']
        assert abs(sum(body['probabilities'].values()) - 1) < 1e-3
        assert body['latency_ms'] >= 0


def test_parity_with_direct_model():
    with TestClient(api.app) as c:
        served = post_image(c).json()['probabilities']
        ck, tf = api.state['ck'], api.state['tf']
        with Image.open(SAMPLE) as img:
            x = tf(img.convert('L' if ck['grayscale'] else 'RGB'))
        with torch.inference_mode():
            direct = F.softmax(api.state['model'](x.unsqueeze(0)), 1)[0]
        for label, probability in zip(ck['classes'], direct.tolist()):
            assert abs(served[label] - probability) < 1e-3


def test_rejects_wrong_type():
    with TestClient(api.app) as c:
        assert c.post('/predict', files={'file': ('a.txt', b'hello', 'text/plain')}).status_code == 415


def test_rejects_too_large():
    with TestClient(api.app) as c:
        big = io.BytesIO(b'0' * (api.MAX_BYTES + 1))
        assert c.post('/predict', files={'file': ('b.jpg', big, 'image/jpeg')}).status_code == 413


def test_rejects_corrupt_image():
    with TestClient(api.app) as c:
        assert c.post('/predict', files={'file': ('c.jpg', b'not an image', 'image/jpeg')}).status_code == 400


def test_missing_file_is_422():
    with TestClient(api.app) as c:
        assert c.post('/predict').status_code == 422


def test_analyze_masks_pii():
    note = 'Fever and fast breath. Guardian 0300-1234567, CNIC 35202-1234567-1, mail person@example.com'
    with TestClient(api.app) as c, open(SAMPLE, 'rb') as image:
        response = c.post('/analyze', files={'file': ('sample.jpeg', image, 'image/jpeg')}, data={'note': note})
        assert response.status_code == 200
        body = response.json()
        assert all(secret not in response.text for secret in ('1234567', '35202', 'person@example.com'))
        assert all(tag in body['note_masked'] for tag in ('[PHONE]', '[CNIC]', '[EMAIL]'))
        assert body['decision'] == 'REFER'
        assert 'fever' in body['urgent_terms']


def test_analyze_clinical_fields_and_mrn_privacy():
    note = 'MRN-204517, temperature 38.5 C, SpO2 91%; battery 91%.'
    with TestClient(api.app) as c, open(SAMPLE, 'rb') as image:
        response = c.post('/analyze', files={'file': ('sample.jpeg', image, 'image/jpeg')}, data={'note': note})
    assert response.status_code == 200
    body = response.json()
    assert '204517' not in response.text
    assert '[MRN]' in body['note_masked']
    assert body['fields']['temperature'] == ['38.5 C']
    assert body['fields']['spo2'] == ['91']


def test_rejects_image_mime_mismatch():
    with TestClient(api.app) as c, open(SAMPLE, 'rb') as image:
        response = c.post('/predict', files={'file': ('sample.png', image, 'image/png')})
    assert response.status_code == 415
