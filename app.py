"""
Simple prediction API for the Titanic survival model.

Run locally:
    pip install fastapi uvicorn joblib scikit-learn pandas
    uvicorn app:app --reload

Then POST to /predict, e.g.:
    curl -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"pclass":3,"sex":1,"age":22,"sibsp":1,"parch":0,"fare":7.25,"alone":0,"embarked_Q":0,"embarked_S":1}'
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survival Prediction API")

model = joblib.load("model.joblib")
feature_columns = joblib.load("feature_columns.joblib")


class PassengerFeatures(BaseModel):
    pclass: int      # 1, 2, or 3
    sex: int         # 0 = female, 1 = male (label-encoded, matches training)
    age: float        # already scaled if you're matching training preprocessing
    sibsp: int        # siblings/spouses aboard
    parch: int        # parents/children aboard
    fare: float        # already scaled if you're matching training preprocessing
    alone: int        # 0 or 1
    embarked_Q: int    # one-hot flag
    embarked_S: int    # one-hot flag


@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running. POST to /predict."}


@app.post("/predict")
def predict(features: PassengerFeatures):
    row = pd.DataFrame([features.dict()])[feature_columns]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "survived_prediction": prediction,
        "survival_probability": round(probability, 3),
    }
