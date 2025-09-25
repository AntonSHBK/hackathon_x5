from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
import os, json
from pathlib import Path

from ml.pipline import NERPipelineCRF
from app.microbatch import MicroBatcher

app = FastAPI(title="X5 NER", version="1.2.0")

class InModel(BaseModel):
    input: str

PIPELINE: NERPipelineCRF | None = None
BATCHER: MicroBatcher | None = None

async def _infer_batch(texts: List[str]) -> List[List[Dict[str, Any]]]:
    assert PIPELINE is not None

    if hasattr(PIPELINE, "predict_batch"):
        out = PIPELINE.predict_batch(texts)
        return out
    results: List[List[Dict[str, Any]]] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            results.append([])
            continue
        ents = PIPELINE.predict_text(t)
        results.append(ents if isinstance(ents, list) else [])
    return results

@app.on_event("startup")
async def startup():
    global PIPELINE, BATCHER
    data_dir = Path(os.getenv("MODEL_DIR", "/app/data/models/"))
    model_dir = data_dir / "ner_x5_88"
    label2idx_path = data_dir / "label2idx.json"
    idx2label_path = data_dir / "idx2label.json"
    max_len = int(os.getenv("MAX_LEN", "128"))

    with open(label2idx_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    with open(idx2label_path, "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}

    PIPELINE = NERPipelineCRF(
        model_path=model_dir,
        label2idx=label2idx,
        idx2label=idx2label,
        max_length=max_len
    )
    max_batch = int(os.getenv("MAX_BATCH", "128"))
    max_wait_ms = int(os.getenv("MAX_WAIT_MS", "8"))
    BATCHER = MicroBatcher(_infer_batch, max_batch=max_batch, max_wait_ms=max_wait_ms)

@app.on_event("shutdown")
async def shutdown():
    if BATCHER:
        await BATCHER.close()

@app.post("/api/predict")
async def predict(body: InModel) -> List[Dict[str, Any]]:
    assert BATCHER is not None
    text = (body.input or "")
    if os.getenv("DISABLE_MICROBATCH") == "1":
        return (await _infer_batch([text]))[0]
    return await BATCHER.submit(text)
