from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, 'saved/model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'saved/scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Dummy user store
USERS = {"admin": "admin123", "demo": "demo123"}

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')
    if USERS.get(username) == password:
        return jsonify({"success": True, "username": username})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodPressure']),
        float(data['skinThickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetesPedigree']),
        float(data['age']),
    ]
    arr = np.array(features).reshape(1, -1)
    arr_s = scaler.transform(arr)
    pred = model.predict(arr_s)[0]
    prob = model.predict_proba(arr_s)[0]
    risk_pct = round(prob[1] * 100, 1)

    risk_level = "Low" if risk_pct < 30 else "Moderate" if risk_pct < 60 else "High"

    factors = []
    if float(data['glucose']) > 140: factors.append("High glucose level")
    if float(data['bmi']) > 30: factors.append("Elevated BMI")
    if float(data['age']) > 45: factors.append("Age above 45")
    if float(data['bloodPressure']) > 90: factors.append("High blood pressure")
    if float(data['insulin']) > 200: factors.append("High insulin")

    return jsonify({
        "prediction": int(pred),
        "probability": risk_pct,
        "riskLevel": risk_level,
        "factors": factors,
        "label": "Diabetic" if pred == 1 else "Non-Diabetic"
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        "totalPredictions": 1284,
        "diabeticCases": 412,
        "nonDiabeticCases": 872,
        "accuracy": 79.4,
        "recentTests": [
            {"id": "P001", "age": 45, "result": "Diabetic",     "risk": 72, "date": "2025-04-06"},
            {"id": "P002", "age": 32, "result": "Non-Diabetic", "risk": 18, "date": "2025-04-06"},
            {"id": "P003", "age": 58, "result": "Diabetic",     "risk": 85, "date": "2025-04-05"},
            {"id": "P004", "age": 27, "result": "Non-Diabetic", "risk": 12, "date": "2025-04-05"},
            {"id": "P005", "age": 51, "result": "Diabetic",     "risk": 63, "date": "2025-04-04"},
        ],
        "monthlyData": [42, 58, 63, 71, 55, 49, 67, 74, 61, 58, 70, 65]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
