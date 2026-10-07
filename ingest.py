import paho.mqtt.client as mqtt
import json
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ---- fill in your token below ----
import os
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
INFLUX_ORG = "digital-twin-org"
INFLUX_BUCKET = "machine_data"
INFLUX_URL = "http://localhost:8086"

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    point = (
        Point("telemetry")
        .field("vibration", data["vibration"])
        .field("temperature", data["temperature"])
    )
    write_api.write(bucket=INFLUX_BUCKET, record=point)
    print("stored:", data)

def on_connect(client, userdata, flags, rc):
    print("connected to MQTT broker, rc =", rc)
    client.subscribe("machine/telemetry")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883)
mqtt_client.loop_forever()