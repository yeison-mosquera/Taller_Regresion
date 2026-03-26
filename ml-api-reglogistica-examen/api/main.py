from fastapi import FastAPI
import pickle
from src.config.settings import MODEL_PATH
from src.models.predict_model import predict

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

@app.post("/predict")
def predict_endpoint(horas: float):
    if model is None:
        return {"error": "Model not loaded"}
    prob = predict(model, horas)
    return {"probabilidad_aprobacion": prob}