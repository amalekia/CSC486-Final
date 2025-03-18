package mqtt;


import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Controller implements ActionListener{
    private Main viewMain;

    public Controller(Main viewMain) {
        this.viewMain = viewMain;
    }

    @Override

    public void actionPerformed(ActionEvent e){
        if (e.getActionCommand().equals("About")) {
            viewMain.about();
        } else if (e.getActionCommand().equals("Start")) {
            viewMain.startEngine();
        } else if (e.getActionCommand().equals("Stop")) {
            viewMain.stopEngine();
        } else if (e.getActionCommand().equals("Connect")) {
            viewMain.startSubscriber();
        } else if (e.getActionCommand().equals("Load")) {
            viewMain.loadFile();
        } else if (e.getActionCommand().equals("Disconnect")) {
            viewMain.stopSubscriber();
        } else if (e.getActionCommand().equals("Switch to Publisher")) {
            viewMain.toggleSubPanel();
        } else if (e.getActionCommand().equals("Switch to Subscriber")) {
            viewMain.toggleSubPanel();
        } else if (e.getActionCommand().equals("Help")) {
            viewMain.help();
        }
    }
}