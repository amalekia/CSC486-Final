import paho.mqtt.client as mqtt

# MQTT broker details (matching the publisher)
MQTT_BROKER = "test.mosquitto.org"  # Broker address (hostname)
MQTT_PORT = 1883  # Broker port
MQTT_TOPIC = "adr-spence"  # Topic to subscribe to
MQTT_CLIENT_ID = "python-subscriber"  # Unique client ID (optional but recommended)

# Callback when the client connects to the MQTT broker
def on_connect(client, userdata, flags, rc, properties=None):  # ADDED properties=None
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

# Callback when a message is received on the subscribed topic
def on_message(client, userdata, msg):
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

def main():
    # Initialize the client with the latest callback API version
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)  # Connect to the broker
    except Exception as e:
        print(f"Connection error: {e}")
        return

    # Start the MQTT client loop to listen for messages
    client.loop_forever()

if __name__ == "__main__":
    main()
