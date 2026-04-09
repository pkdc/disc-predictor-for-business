import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings
from utils.disc_labels import decode_disc_labels

# Restrict TensorFlow memory usage
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

MIN_WORD_COUNT = 10
FORWARD_PATTERN = re.compile(
    r"-{3,}\s*(Original Message|Forwarded by|Forwarded Message)\s*-{0,}",
    re.IGNORECASE,
)

ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    global ml_model
    ml_model = load_model()
    yield


app = FastAPI(
    title="DISC Personality Predictor",
    description="Predicts DISC personality style from business email text.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    disc_labels: list[str]
    warning: str | None = None


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned_text = clean_msg_body(req.text)

    word_count = len(cleaned_text.split())
    if word_count < MIN_WORD_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Text too short after cleaning ({word_count} words). Provide at least {MIN_WORD_COUNT} words.",
        )

    has_forward = bool(FORWARD_PATTERN.search(req.text))

    embeddings = get_bert_embeddings([cleaned_text])
    result = ml_model.predict(embeddings)
    disc_labels = list(decode_disc_labels(result)[0])

    return PredictResponse(
        disc_labels=disc_labels,
        warning="This appears to be a forwarded/reply email. Results may be less accurate." if has_forward else None,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
