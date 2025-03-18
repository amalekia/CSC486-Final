package mqtt;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;

import java.util.List;


public class Publisher implements AutoCloseable, Runnable {
	private String csvFile;


	private static final String BROKER = "tcp://test.mosquitto.org:1883";
	private static final String TOPIC = "adr-spence";
	private static final String CLIENT_ID = "csv-publisher";
	private MqttClient client;

	private boolean wait = true;
	private boolean stop = false;

	public void stop(boolean stop) {
		this.stop = stop;
	}
	public Publisher(String filePath) throws MqttException {
		this.csvFile = filePath;
		client = new MqttClient(BROKER, CLIENT_ID);
		client.connect();
		System.out.println("Connected to BROKER: " + BROKER);
	}

	public void publishMessage(String message) throws MqttException {

		MqttMessage mqttMessage = new MqttMessage(message.getBytes());

		mqttMessage.setQos(2); // Quality of Service

		client.publish(TOPIC, mqttMessage);

		System.out.println("Message published: " + message);

	}


	@Override
	public void close() throws MqttException {
		if (client != null && client.isConnected()) {
			client.disconnect();
			System.out.println("Disconnected from BROKER: " + BROKER);
		}
	}

	@Override
	public void run() {
		//String csvFile = "/Users/sammorrisroe/Desktop/CSC486_MQTT/csv-subscriber-tcptestmosquittoorg1883/s-1.msg.csv";

      try {
         List<String> csvLines = Files.readAllLines(Paths.get(csvFile));

         System.out.println("Publishing messages from CSV...");

         // Publish each line with a 1-second delay
         int lineCounter = 0;
         for (String line : csvLines) {
            // Add lines in csv to Repo ArrayList
            lineCounter++;
            Repository.getInstance().addData(String.valueOf(lineCounter), line);
            this.publishMessage(line);
            Thread.sleep(1000); // 1-second delay
         }

         Thread.sleep(1000); // 1-second delay

         System.out.println("Finished publishing all messages.");
      } catch (MqttException | InterruptedException | IOException e) {
         e.printStackTrace();
      }
   }
}
