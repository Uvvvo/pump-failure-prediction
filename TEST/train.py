import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

df = pd.read_csv("centrifugal_pump_readings.csv")
df = df.sort_values("timestamp").reset_index(drop=True)

features = ["Temperature", "Pressure", "Flow_Rate", "RPM", "Vibration", "Power"]
target = "Maintenance_Flag"

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, "scaler_centrifugal.pkl")

df["Future_Flag"] = df["Maintenance_Flag"].shift(-5).fillna(0)
df["Future_Flag"] = df["Future_Flag"].astype(int)

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X, y = create_sequences(df[features].values, df["Future_Flag"].values, time_steps=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(cw))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    class_weight=cw,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

model.save("centrifugal_pump_failure_model.h5")
print("\n💾 Model saved successfully as centrifugal_pump_failure_model.h5")