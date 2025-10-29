import pandas as pd
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model

model = load_model("centrifugal_pump_failure_model.h5")
scaler = joblib.load("scaler_centrifugal.pkl")

features = ["Temperature", "Pressure", "Flow_Rate", "RPM", "Vibration", "Power"]
window = []

data = pd.read_csv("pump_live_data.csv")

print("🧠 بدأ التحليل الواقعي للقراءات...\n")

for i in range(len(data)):
    row = data.iloc[i][features].values
    window.append(row)

    if len(window) > 60:
        window = window[-60:]

    if len(window) == 60:
        X = np.array(window).reshape(1, 60, len(features))
        X_scaled = scaler.transform(np.array(window))
        X_scaled = X_scaled.reshape(1, 60, len(features))
        pred = model.predict(X_scaled)[0][0]

        state = "✅ Normal" if pred < 0.5 else "⚠️ Warning"
        print(f"{data.iloc[i]['timestamp']} | {state} | Risk: {pred*100:.1f}%")

    time.sleep(0.2)  # سرعة عرض البيانات (تقدر تخليها أبطأ)