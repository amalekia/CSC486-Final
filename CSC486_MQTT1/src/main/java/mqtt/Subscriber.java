package mqtt;

import org.eclipse.paho.client.mqttv3.*;
import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This class is a simple MQTT subscriber that listens to a TOPIC.
 * The BROKER is test.mosquitto.org and the TOPIC is cal-poly/csc/309.
 * (run this and the publisher at the same time)
 *
 * NOTE: Change the file path to CSV before running
 *
 * @author javiergs Adrick Malekian Spencer Perley
 * @version 1.0
 */
public class Subscriber implements MqttCallback, Runnable {
	public static String line;
	public int counter = 0;
	private final static String BROKER = "tcp://test.mosquitto.org:1883";
	private final static String TOPIC = "adr-spence";
	private final static String CLIENT_ID = "csv-subscriber";
	private MqttClient client;


	private boolean stop = false;

	public void stop(boolean stop) {
		this.stop = stop;
	}

	@Override
	public void run() {
		try {
			client = new MqttClient(BROKER, CLIENT_ID);
			client.setCallback(this); // Set this class as the callback handler
			client.connect();
			System.out.println("Connected to BROKER: " + BROKER);
			client.subscribe(TOPIC);
			System.out.println("Subscribed to TOPIC: " + TOPIC);
		} catch (MqttException e) {
			e.printStackTrace();
		}
	}

	public void closeSub() throws MqttException {
		if (client != null && client.isConnected()) {
			client.disconnect();
			System.out.println("Disconnected from BROKER: " + BROKER);
		}
	}
	
	@Override
	public void connectionLost(Throwable throwable) {
		System.out.println("Connection lost: " + throwable.getMessage());
	}

	@Override
	public void messageArrived(String topic, MqttMessage mqttMessage) {
		String message = new String(mqttMessage.getPayload());
		System.out.println("Received message on topic: " + topic + " -> " + message);
		line = message; // Store if needed
	}

	
	@Override
	public void deliveryComplete(IMqttDeliveryToken iMqttDeliveryToken) {
		System.out.println("Delivered complete: " + iMqttDeliveryToken.getMessageId());
	}

}
