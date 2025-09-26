# --- сверху файла ---
import os, json, time, logging, asyncio
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel

from ml.pipline import NERPipelineCRF
from app.microbatch import MicroBatcher

log = logging.getLogger("perf")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="X5 NER", version="1.3.0")

class InModel(BaseModel):
    input: str

PIPELINE: NERPipelineCRF | None = None
BATCHER: MicroBatcher | None = None
_INIT_LOCK = asyncio.Lock()   # защита от гонки при одновременной первой инициализации

async def _ensure_inited():
    """Ленивая инициализация пайплайна и микробатчера (потокобезопасно)."""
    global PIPELINE, BATCHER
    if PIPELINE is not None and BATCHER is not None:
        return
    async with _INIT_LOCK:
        if PIPELINE is not None and BATCHER is not None:
            return
        # --- загрузка модели ---
        data_dir = Path(os.getenv("MODEL_DIR", "/app/data/models/"))
        model_dir = data_dir / "ner_x5_tiny_89"
        label2idx_path = data_dir / "label2idx.json"
        idx2label_path = data_dir / "idx2label.json"
        max_len = int(os.getenv("MAX_LEN", "128"))

        # читаем метаданные
        with open(label2idx_path, "r", encoding="utf-8") as f:
            label2idx = json.load(f)
        with open(idx2label_path, "r", encoding="utf-8") as f:
            idx2label = {int(k): v for k, v in json.load(f).items()}

        PIPELINE = NERPipelineCRF(
            model_path=str(model_dir),
            label2idx=label2idx,
            idx2label=idx2label,
            max_length=max_len
        )

        # микробатчер
        max_batch = int(os.getenv("MAX_BATCH", "128"))
        max_wait_ms = int(os.getenv("MAX_WAIT_MS", "8"))
        BATCHER = MicroBatcher(_infer_batch, max_batch=max_batch, max_wait_ms=max_wait_ms)

        # прогрев (не обязательно, но полезно)
        try:
            _ = await _infer_batch(["йогурт даниссимо 0.5л 15%"])
        except Exception as e:
            log.warning("Warmup failed: %s", e)

async def _infer_batch(texts: List[str]) -> List[List[Dict[str, Any]]]:
    await _ensure_inited()
    assert PIPELINE is not None

    # отбрасываем пустые — вернём [] на их места
    idx_map, clean = [], []
    for i, t in enumerate(texts):
        t = (t or "").strip()
        if t:
            idx_map.append(i)
            clean.append(t)
    if not clean:
        return [[] for _ in texts]

    res = PIPELINE.predict(clean)

    out = [[] for _ in texts]
    for i_src, r in zip(idx_map, res):
        out[i_src] = r
    return out

@app.post("/api/predict")
async def predict(body: InModel) -> List[Dict[str, Any]]:
    # await _ensure_inited()
    if os.getenv("DISABLE_MICROBATCH") == "1":
        return (await _infer_batch([body.input]))[0]
    assert BATCHER is not None
    return await BATCHER.submit(body.input)
