import pickle
import pandas as pd
from fastapi import FastAPI
from typing import Any
import uvicorn


app = FastAPI(title='predict-gpa')


with open('artifacts/xgb_model.pkl', 'rb') as f:
    preprocessor, model = pickle.load(f)


@app.get('/health')
def health():
    return {'status': 'OK'}


@app.post('/predict')
def predict(record: dict[str, Any]):
    record_clean = {
        'study_hours_per_day': record.get('study_hours_per_day', 0),
        'extracurricular_hours_per_day': record.get('extracurricular_hours_per_day', 0),
        'sleep_hours_per_day': record.get('sleep_hours_per_day', 0),
        'social_hours_per_day': record.get('social_hours_per_day', 0),
        'physical_activity_hours_per_day': record.get('physical_activity_hours_per_day', 0),
        'stress_level': record.get('stress_level', 'Unknown'),
    }

    X = pd.DataFrame(
        preprocessor.transform(pd.DataFrame([record_clean])),
        columns=[f.lower() for f in preprocessor.get_feature_names_out()]
    )

    prediction = float(model.predict(X)[0])

    print(f'{record=}: {prediction=}')

    return {'prediction': prediction}
