"""Week 8 API for the Week 6 pediatric X-ray model and Week 7 text evidence."""
import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

MODEL_PATH = os.getenv('MODEL_PATH', 'week6_model.pt')
POSITIVE = os.getenv('POSITIVE', 'PNEUMONIA')
# Week 6 selected this threshold on validation, not on the test set.
THRESHOLD = float(os.getenv('THRESHOLD', '0.8331834077835083'))
API_KEY = os.getenv('API_KEY')
MAX_BYTES = 5 * 1024 * 1024
URGENT = ['fever', 'breath', 'cough', 'chest pain', 'spreading', 'crack']
PATTERNS = {
    'cnic': r'\b\d{5}-\d{7}-\d\b',
    'mobile': r'(?:\+92[- ]?|\b0)3\d{2}[- ]?\d{7}\b',
    'email': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'mrn': r'\bMRN[- :]*\d+\b',
    'temperature': r'\b(?:3[5-9]|4[0-2])(?:\.\d)?\s?°?C\b',
    'spo2': r'\bSpO2\s?[:=]?\s?(\d{2,3})\s?%',
    'amount': r'(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)',
    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
}
state = {}


def rebuild(ck):
    name, k = ck['backbone'], len(ck['classes'])
    if name.startswith('resnet'):
        model = getattr(models, name)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, k)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, k)
    else:
        raise ValueError(f'Unsupported checkpoint backbone: {name}')
    model.load_state_dict(ck['state_dict'])
    return model.eval()


@asynccontextmanager
async def lifespan(app):
    ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    if POSITIVE not in ck['classes']:
        raise ValueError(f'{POSITIVE!r} is not a checkpoint class')
    if not 0 <= THRESHOLD <= 1:
        raise ValueError('THRESHOLD must be between 0 and 1')
    state['ck'], state['model'] = ck, rebuild(ck)
    state['tf'] = transforms.Compose(
        ([transforms.Grayscale(3)] if ck['grayscale'] else []) +
        [transforms.Resize((ck['img_size'],) * 2), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    try:
        yield
    finally:
        state.clear()


app = FastAPI(title='Project 2 Vision + Text API', version='1.0', lifespan=lifespan)


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    latency_ms: float


class CaseRecord(BaseModel):
    prediction: Prediction
    note_masked: str
    fields: dict[str, list[str]]
    urgent_terms: list[str]
    decision: str
    reasons: list[str]
    disclaimer: str


def check_key(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, 'Invalid or missing API key')


async def read_image(file):
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(415, 'Upload a JPEG or PNG image')
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image larger than 5 MB')
    try:
        image = Image.open(io.BytesIO(data))
        if image.format != {'image/jpeg': 'JPEG', 'image/png': 'PNG'}[file.content_type]:
            raise HTTPException(415, 'Upload a JPEG or PNG image')
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, 'File is not a readable image')


def classify(image):
    ck, start = state['ck'], time.perf_counter()
    x = state['tf'](image.convert('L' if ck['grayscale'] else 'RGB'))
    with torch.inference_mode():
        probabilities = F.softmax(state['model'](x.unsqueeze(0)), dim=1)[0].tolist()
    probs = dict(zip(ck['classes'], probabilities))
    return Prediction(label=max(probs, key=probs.get), probabilities=probs,
                      latency_ms=round((time.perf_counter() - start) * 1000, 1))


def mask_pii(text):
    for key, tag in [('cnic', '[CNIC]'), ('mobile', '[PHONE]'), ('email', '[EMAIL]'), ('mrn', '[MRN]')]:
        text = re.sub(PATTERNS[key], tag, text, flags=re.IGNORECASE)
    return text


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in state,
            'classes': state['ck']['classes'], 'positive': POSITIVE}


@app.post('/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    return await run_in_threadpool(classify, await read_image(file))


@app.post('/analyze', response_model=CaseRecord)
async def analyze(file: UploadFile = File(...), note: str = Form(...),
                  x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    if len(note) > 2000:
        raise HTTPException(422, 'Note longer than 2,000 characters')
    pred = await run_in_threadpool(classify, await read_image(file))
    safe = mask_pii(note)
    fields = {k: re.findall(p, safe, flags=re.IGNORECASE) for k, p in PATTERNS.items()}
    urgent = [word for word in URGENT if word in safe.lower()]
    reasons = []
    if pred.probabilities[POSITIVE] >= THRESHOLD:
        reasons.append(f'image: P({POSITIVE}) = {pred.probabilities[POSITIVE]:.2f}')
    if urgent:
        reasons.append('note mentions: ' + ', '.join(urgent))
    return CaseRecord(prediction=pred, note_masked=safe,
                      fields={k: v for k, v in fields.items() if v},
                      urgent_terms=urgent, decision='REFER' if reasons else 'ROUTINE',
                      reasons=reasons,
                      disclaimer='Academic decision support only; not for clinical use.')
