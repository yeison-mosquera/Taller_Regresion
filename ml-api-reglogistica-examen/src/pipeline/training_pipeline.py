from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
import pickle
from src.config.settings import MODEL_PATH

def run_pipeline():
    df = load_data()
    X, y = preprocess(df)
    X = build_features(X)
    model = train_model(X, y)
    score = evaluate_model(model, X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained with accuracy: {score}")
    return model