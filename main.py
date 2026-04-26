from fastapi import FastAPI
from pydantic import BaseModel, Field

from utils.model_loader import load_model
from utils.preprocess import clean_msg_body
from utils.bert_embedder import get_bert_embeddings
from utils.disc_labels import decode_disc_labels

model = load_model()

app = FastAPI()

MAX_INPUT_LENGTH = 50000  # ~35,000 words — well above any realistic email

class PredictRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=MAX_INPUT_LENGTH)

class PredictResponse(BaseModel):
    disc_labels: list[str]

@app.post("/predict")
def predict_disc_personality(req: PredictRequest):
    cleaned_text = clean_msg_body(req.input_text)
    embeddings = get_bert_embeddings([cleaned_text])
    result = model.predict(embeddings)
    disc_labels = decode_disc_labels(result)
    return PredictResponse(disc_labels=disc_labels)