"""Benchmark a running server: python bench.py (50 sequential requests)."""
import os
import time

import httpx
import numpy as np

url = os.getenv('API_URL', 'http://127.0.0.1:8000/predict')
sample = os.getenv('SAMPLE_IMAGE', 'sample.jpeg')
headers = {'X-API-Key': os.environ['API_KEY']} if os.getenv('API_KEY') else {}
with open(sample, 'rb') as image:
    data = image.read()
latencies = []
with httpx.Client(timeout=30, headers=headers) as client:
    for _ in range(50):
        start = time.perf_counter()
        response = client.post(url, files={'file': ('sample.jpeg', data, 'image/jpeg')})
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)
print(f'n={len(latencies)} p50={np.percentile(latencies, 50):.1f} ms '
      f'p95={np.percentile(latencies, 95):.1f} ms max={max(latencies):.1f} ms')
print(f'Sequential throughput ~ {1000 / np.mean(latencies):.1f} requests/s')
