---
title: CardioScan Inference
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cardiomegaly screening API for the CardioScan frontend.
---

# CardioScan Inference

FastAPI service that serves the chest X-ray cardiomegaly ensemble trained in
[heart-scan-helper](https://github.com/) — a torchxrayvision DenseNet-121
3-seed ensemble with optional 6-pass test-time augmentation.

## Endpoints

- `GET  /health` &nbsp;— readiness + model metadata
- `POST /predict` &nbsp;— `multipart/form-data`, field name `image`
- `POST /debug/predict` &nbsp;— per-model, per-TTA logits (debugging)

Response shape:

```json
{
  "prediction": "Cardiomegaly",
  "confidence": 0.873,
  "heatmap_url": null,
  "source": "model",
  "threshold": 0.504486,
  "ensemble_size": 3,
  "use_tta": true
}
```

## Environment

CORS already allows `*.lovable.app`, `*.lovableproject.com`, `*.hf.space` and
`localhost`. Override with `ALLOWED_ORIGIN_REGEX` if you need to lock it down.
