from sklearn.externals import joblib
from keras.models import load_model
import numpy as np

def predict(model_file, X):
    model = joblib.load(model_file)
    return model.predict(X)

def predict_deep(model_file, X):
    model = load_model(model_file)
    y_proba = model.predict(X)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred
