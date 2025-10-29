import pandas as pd
import numpy as np
import datetime
import time

n_points = 500  # عدد القراءات
fault_start = 400  # بعد هاي القراءة يبدي العطل

records = []

for i in range(n_points):
    timestamp = datetime.datetime.now() + datetime.timedelta(minutes=i*5)

    if i < fault_start:
        temp = np.random.normal(80, 1.5)
        pressure = np.random.normal(3.5, 0.2)
        vibration = np.random.normal(0.5, 0.1)
    else:
        temp = np.random.normal(100, 3.0)
        pressure = np.random.normal(5.0, 0.3)
        vibration = np.random.normal(3.0, 0.5)

    flow = np.random.normal(150, 2)
    rpm = np.random.normal(2900, 10)
    power = np.random.normal(20 + (temp - 80) * 0.5, 1)

    records.append({
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature": temp,
        "Pressure": pressure,
        "Flow_Rate": flow,
        "RPM": rpm,
        "Vibration": vibration,
        "Power": power
    })

df = pd.DataFrame(records)
df.to_csv("pump_live_data.csv", index=False)
print("✅ تم توليد القراءات بنجاح في ملف pump_live_data.csv")