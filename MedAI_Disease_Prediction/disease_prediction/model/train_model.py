import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# ─── Generate Synthetic Diabetes Dataset ───────────────────────────────────────
np.random.seed(42)
n = 800

data = {
    'Pregnancies':        np.random.randint(0, 17, n),
    'Glucose':            np.random.randint(70, 200, n),
    'BloodPressure':      np.random.randint(40, 122, n),
    'SkinThickness':      np.random.randint(0, 99, n),
    'Insulin':            np.random.randint(0, 846, n),
    'BMI':                np.round(np.random.uniform(18.0, 67.1, n), 1),
    'DiabetesPedigree':   np.round(np.random.uniform(0.08, 2.42, n), 3),
    'Age':                np.random.randint(21, 81, n),
}

df = pd.DataFrame(data)

# Simulate realistic outcome
score = (
    (df['Glucose'] > 140).astype(int) * 2 +
    (df['BMI'] > 30).astype(int) * 1.5 +
    (df['Age'] > 45).astype(int) * 1 +
    (df['Pregnancies'] > 5).astype(int) * 0.5 +
    np.random.normal(0, 1, n)
)
df['Outcome'] = (score > 2.5).astype(int)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

acc = accuracy_score(y_test, model.predict(X_test_s))
print(f"Model Accuracy: {acc*100:.2f}%")

os.makedirs('saved', exist_ok=True)
with open('saved/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('saved/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved to saved/")
