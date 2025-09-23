from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import os, json
from pathlib import Path

# импортируем твой пайплайн
from ml.pipline import NERPipelineCRF

app = FastAPI(title="X5 NER Service", version="1.0.0")

class InModel(BaseModel):
    input: str

@app.on_event("startup")
async def load_model():
    data_dir = Path(os.getenv("MODEL_DIR", "/app/data/models/"))
    model_dir = data_dir / "ner_x5_88"
    label2idx_path = data_dir / "label2idx.json"
    idx2label_path = data_dir / "idx2label.json"

    with open(label2idx_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)

    with open(idx2label_path, "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}

    pipeline = NERPipelineCRF(
        model_path=model_dir,
        label2idx=label2idx,
        idx2label=idx2label,
        max_length=int(os.getenv("MAX_LEN", "128"))
    )

    app.state.pipeline = pipeline

@app.post("/api/predict")
async def predict(body: InModel) -> List[Dict[str, Any]]:
    text = (body.input or "").strip()
    if not text:
        return []

    pipeline: NERPipelineCRF = app.state.pipeline
    entities = pipeline.predict_text(text)

    return entities
