import threading
import time
import subprocess

def run_generator():
    subprocess.run(["python", "generator.py"])

def run_monitor():
    time.sleep(20)  # ننتظر شوية حتى يولّد أول قراءات
    subprocess.run(["python", "monitor.py"])

t1 = threading.Thread(target=run_generator)
t2 = threading.Thread(target=run_monitor)

t1.start()
t2.start()