package mqtt;

import javax.swing.*;
import java.awt.*;

public class Main extends JFrame {
    private Thread publisherThread;
    private Thread subscriberThread;
    private boolean engineRunning = false;
    private boolean subRunning = false;
    private Publisher publisher;
    private Subscriber subscriber;
    private ViewPanel centralPanel;
    private String selectedFilePath = null;
    public static boolean isSubscriber = false;

    public Main() {
        setJMenuBar(createPubMenuBar());

        centralPanel = new ViewPanel();
        centralPanel.setVisible(false);
        StatusBar viewStatusBar = new StatusBar();

        setLayout(new BorderLayout());
        add(centralPanel, BorderLayout.CENTER);
        add(viewStatusBar, BorderLayout.SOUTH);

        Repository.getInstance().pcs.addPropertyChangeListener(centralPanel);
        Repository.getInstance().pcs.addPropertyChangeListener(viewStatusBar);
    }


    public void loadFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select CSV File");
        int result = fileChooser.showOpenDialog(this);

        if (result == JFileChooser.APPROVE_OPTION) {
            selectedFilePath = fileChooser.getSelectedFile().getAbsolutePath();
            JOptionPane.showMessageDialog(this, "File Loaded: " + selectedFilePath);
        }
    }

    public void toggleSubPanel() {
        if (!isSubscriber) {
            setJMenuBar(createSubMenuBar());
            centralPanel.setVisible(true);
            setTitle("Subscriber");
            this.isSubscriber = true;
        } else {
            setJMenuBar(createPubMenuBar());
            centralPanel.setVisible(false);
            setTitle("Publisher");
            this.isSubscriber = false;
        }
        revalidate();
        repaint();
    }

    private JMenuBar createSubMenuBar() {
        Controller controller = new Controller(this);

        JMenuBar menuBar = new JMenuBar();
        JMenu serverMenu = new JMenu("Server");
        JMenu aboutMenu = new JMenu("About");
        JMenu helpMenu = new JMenu("Help");
        JMenuItem connectItem = new JMenuItem("Connect");
        JMenuItem disconnectItem = new JMenuItem("Disconnect");
        JMenuItem aboutItem = new JMenuItem("About");
        JMenuItem helpItem = new JMenuItem("Help");
        JMenuItem toggleModeItem = new JMenuItem("Switch to Publisher");

        serverMenu.add(connectItem);
        serverMenu.add(disconnectItem);
        aboutMenu.add(aboutItem);
        helpMenu.add(helpItem);

        toggleModeItem.addActionListener(controller);
        connectItem.addActionListener(controller);
        disconnectItem.addActionListener(controller);
        aboutItem.addActionListener(controller);
        helpItem.addActionListener(controller);

        menuBar.add(serverMenu);
        menuBar.add(aboutMenu);
        menuBar.add(helpMenu);
        menuBar.add(toggleModeItem);
        return menuBar;
    }

    public JMenuBar createPubMenuBar() {
        Controller controller = new Controller(this);

        JMenuBar menuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("File");
        JMenu serverMenu = new JMenu("Server");
        JMenu aboutMenu = new JMenu("About");
        JMenuItem loadItem = new JMenuItem("Load");
        JMenuItem startItem = new JMenuItem("Start");
        JMenuItem stopItem = new JMenuItem("Stop");
        JMenuItem aboutItem = new JMenuItem("About");
        JMenuItem toggleModeItem = new JMenuItem("Switch to Subscriber");

        startItem.addActionListener(controller);
        stopItem.addActionListener(controller);
        toggleModeItem.addActionListener(controller);

        fileMenu.add(loadItem);
        serverMenu.add(startItem);
        serverMenu.add(stopItem);
        aboutMenu.add(aboutItem);

        loadItem.addActionListener(controller);
        aboutItem.addActionListener(controller);

        menuBar.add(fileMenu);
        menuBar.add(serverMenu);
        menuBar.add(aboutMenu);
        menuBar.add(toggleModeItem);
        return menuBar;
    }

    public void help() {
        JOptionPane.showMessageDialog(this, "Connect: Establishes a connection to the broker\n" +
                "Disconnect: Disconnects a connection to the broker");
    }

    public void startEngine() {
        if (!engineRunning) {
            if (selectedFilePath == null) {
                JOptionPane.showMessageDialog(this, "Please load a CSV file first!");
                return;
            }

            try {
                publisher = new Publisher(selectedFilePath);
                publisherThread = new Thread(publisher);
                publisherThread.start();
                engineRunning = true;
                JOptionPane.showMessageDialog(this, "Publisher started with file: " + selectedFilePath);
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Failed to start Publisher: " + e.getMessage());
            }
        } else {
            JOptionPane.showMessageDialog(this, "Engine (Publisher) is already running.");
        }
    }

    public void stopEngine() {
        if (engineRunning) {
            try {
                publisher.stop(true);  // Signal to stop
                if (publisherThread != null) {
                    publisherThread.interrupt();
                    publisherThread.join();
                }
                publisher.close();
                engineRunning = false;
                JOptionPane.showMessageDialog(this, "Engine (Publisher) stopped.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error stopping Publisher: " + e.getMessage());
            }
        } else {
            JOptionPane.showMessageDialog(this, "Engine is not running.");
        }
    }

    public void startSubscriber() {
        if (!subRunning) {
            try {
                subscriber = new Subscriber();
                subscriberThread = new Thread(subscriber);
                subscriberThread.start();
                subRunning = true;
                JOptionPane.showMessageDialog(this, "Engine (Subscriber) started.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Failed to start Subscriber: " + e.getMessage());
            }
        } else {
            JOptionPane.showMessageDialog(this, "Engine (Subscriber) is already running.");
        }
    }

    public void stopSubscriber() {
        Subscriber.line = null;
        if (subRunning) {
            try {
                subscriber.stop(true);  // Signal to stop
                if (subscriberThread != null) {
                    subscriberThread.interrupt();
                    subscriberThread.join();
                }
                subscriber.closeSub();
                subRunning = false;
                JOptionPane.showMessageDialog(this, "Engine (Subscriber) stopped.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error stopping Subscriber: " + e.getMessage());
            }
        } else {
            JOptionPane.showMessageDialog(this, "Subscriber is not running.");
        }
    }

    public void about() {
        JOptionPane.showMessageDialog(this, "Sam Morrisroe, Spencer Perley, Adrick Malekian");
    }

    public static void main(String[] args) {
        Main frame = new Main();
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        frame.setTitle("Publisher");
    }
}
