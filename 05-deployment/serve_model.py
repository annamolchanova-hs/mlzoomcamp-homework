import pickle
from fastapi import FastAPI
from typing import Any
import uvicorn


app = FastAPI(title='make-prediction')


with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)


@app.post('/predict')
def predict(record: dict[str, Any]):
    prob = float(pipeline.predict_proba(record)[0, 1])

    return {'prediction': prob}
