import paho.mqtt.client as mqtt
import json
import time
import random

# USE THE SAME BROKER AS APP.PY
BROKER = 'test.mosquitto.org' 
PORT = 1883
TOPIC = 'iot-p2p-trading/trades'

def publish_trade():
    client = mqtt.Client()
    
    print(f"Connecting to {BROKER}...")
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Simulate random trade data
    trade_data = {
        "buyer": "HH002",
        "seller": "HH001",
        "qty_kwh": round(random.uniform(1.0, 5.0), 2),
        "price": round(random.uniform(5.0, 8.0), 2),
        "timestamp": time.time()
    }
    
    payload = json.dumps(trade_data)
    client.publish(TOPIC, payload)
    print(f"✅ Sent message: {payload}")
    
    client.disconnect()

if __name__ == "__main__":
    # Send 5 messages to test
    for _ in range(5):
        publish_trade()
        time.sleep(2)