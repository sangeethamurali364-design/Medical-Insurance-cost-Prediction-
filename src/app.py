from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/best_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred_log = model.predict(df)
    pred = np.expm1(pred_log)
    return jsonify({'predicted_charges': float(pred[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
