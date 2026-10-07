import paho.mqtt.client as mqtt
import time, random, json

client = mqtt.Client()
client.connect("localhost", 1883)
t = 0
fault_mode = False

while True:
    t += 1
    if t % 200 == 0:
        fault_mode = not fault_mode

    if fault_mode:
        vibration = round(random.uniform(6.0, 9.0), 2)
        temperature = round(random.uniform(85, 100), 2)
    else:
        vibration = round(random.uniform(1.0, 3.0), 2)
        temperature = round(random.uniform(40, 60), 2)

    payload = json.dumps({"t": t, "vibration": vibration, "temperature": temperature})
    client.publish("machine/telemetry", payload)
    print("sent:", payload)
    time.sleep(1)