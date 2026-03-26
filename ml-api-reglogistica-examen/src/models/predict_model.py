def predict(model, horas):
    return model.predict_proba([[horas]])[0][1]