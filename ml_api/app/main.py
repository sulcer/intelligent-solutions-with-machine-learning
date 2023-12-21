import pickle
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from app.utils.sliding_window import sliding_window
import numpy as np


class PredictionInput(BaseModel):
    temperature: float
    apparent_temperature: float
    precipitation_probability: float
    hour: int
    apparent_temperature_difference: float


class SecondPredictionInput(BaseModel):
    date: str
    temperature: float
    relative_humidity: float
    dew_point: float
    apparent_temperature: float
    precipitation_probability: float
    rain: float
    surface_pressure: float
    bike_stands: int
    available_bike_stands: int


# Neural Network Model
nn_model = pickle.load(open('app/models/nn/regressor.pkl', 'rb'))
nn_scaler = pickle.load(open('app/models/nn/scaler.pkl', 'rb'))


# Recurrent Neural Network Model
rnn_model = load_model('app/models/rnn/model_gru.h5')
rnn_scaler = pickle.load(open('app/models/rnn/scaler.pkl', 'rb'))


app = FastAPI()


@app.get("/health")
async def health():
    return {
        "status": "api is up and running",
    }


@app.post("/predict/NN_model")
async def predict(data: PredictionInput):
    input_features = [
        data.apparent_temperature_difference,
        data.apparent_temperature,
        data.temperature,
        data.hour,
        data.precipitation_probability
    ]

    features = np.array(input_features).reshape(1, -1)
    features = nn_scaler.transform(features)

    prediction = nn_model.predict(features)
    return {"prediction": prediction.item()}


@app.post("/predict/RNN_model")
async def predict_line(data: List[SecondPredictionInput]):
    data = [d.dict() for d in data]
    data = sorted(data, key=lambda x: x['date'])
    data = [d['available_bike_stands'] for d in data]
    data = np.array(data).reshape(-1, 1)
    data = rnn_scaler.transform(data)

    X, y_test = sliding_window(data, 186)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    prediction = rnn_model.predict(X)
    prediction = rnn_scaler.inverse_transform(prediction)
    prediction = prediction.flatten().tolist()

    return {"prediction": prediction}
