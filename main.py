from pydantic import BaseModel
from fastapi import FastAPI

import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(
        title="Titanic Survival Prediction API",
        description="An API for predicting Titanic survival using a trained Logistic Regression model.",
        version="1.0.0",
)

model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    Age: float
    Sex: int
    Pclass: int 

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the request data to a pandas DataFrame
    df = pd.DataFrame([request.model_dump()])

    # Make a prediction using the loaded model
    prediction = model.predict(df)

    # Return the prediction
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)