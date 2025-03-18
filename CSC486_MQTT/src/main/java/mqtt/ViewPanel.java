package mqtt;
import javax.swing.*;
import java.awt.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.HashMap;
import java.util.Map;

public class ViewPanel extends JPanel implements PropertyChangeListener {

    public JTextArea textArea;
    private Map<String, JPanel> emotionPanels;

    public ViewPanel() {
        setLayout(new GridLayout(1,2));
        textArea = new JTextArea();
        textArea.setEditable(false);

// Emotion panel grid
        JPanel emotionGrid = new JPanel(new GridLayout(2, 4, 5, 5));
        emotionPanels = new HashMap<>();

        String[] emotions = {"+++","++-","+-+","+--","-++","-+-","--+","---"};
        for (String emotion : emotions) {
            JPanel panel = createEmotionPanel(emotion);
            emotionPanels.put(emotion, panel);
            emotionGrid.add(panel);
        }

        add(new JScrollPane(textArea), BorderLayout.NORTH);
        add(emotionGrid, BorderLayout.CENTER);
    }

    private JPanel createEmotionPanel(String text) {
        JPanel panel = new JPanel();
        panel.setBackground(Color.WHITE);
        panel.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2));
        panel.setPreferredSize(new Dimension(80, 80));
        panel.setLayout(new BorderLayout());

        JLabel label = new JLabel(text, SwingConstants.CENTER);
        label.setFont(new Font("Arial", Font.BOLD, 14));
        panel.add(label, BorderLayout.CENTER);

        return panel;
    }
    public void changePanelColor(String emotion, Color newColor) {
        JPanel panel = emotionPanels.get(emotion);
        if (panel != null) {
            panel.setBackground(newColor);
            panel.repaint(); // Ensure UI updates properly
        }
    }


    public void propertyChange(PropertyChangeEvent evt) {
        if (Subscriber.line != null) {
            textArea.append(Subscriber.line + "\n");
            String[] words = Subscriber.line.split(", ");
            String pleasure = String.join(" ","Pleasure", words[26]);
            String arousal = String.join(" ","Arousal", words[27]);
            String dominance = String.join(" ","Dominance", words[28]);
            String finalStr = String.join(", ", pleasure, arousal, dominance);
            textArea.append(finalStr + "\n");
            updateEmotions(Float.parseFloat(words[26]),Float.parseFloat(words[27]),Float.parseFloat(words[28]));

        }

    }

    public void updateEmotions(float P, float A, float D){

        if (P > 0.5 && A > 0.5 && D > 0.5){
            //+++
            changePanelColor("+++", Color.GREEN);
        }else{
            changePanelColor("+++", Color.WHITE);
        }

        if (P > 0.5 && A > 0.5 && D < 0.5){
            //++-
            changePanelColor("++-", Color.GREEN);

        }else{
            changePanelColor("++-", Color.WHITE);
        }

        if (P > 0.5 && A < 0.5 && D > 0.5){
            //+-+
            changePanelColor("+-+", Color.GREEN);

        }else{
            changePanelColor("+-+", Color.WHITE);
        }

        if (P > 0.5 && A < 0.5 && D < 0.5) {
            //+--
            changePanelColor("+--", Color.GREEN);

        }else{
            changePanelColor("+--", Color.WHITE);
        }

        if(P < 0.5 && A > 0.5 && D > 0.5){
            //-++
            changePanelColor("-++", Color.GREEN);

        }else{
            changePanelColor("-++", Color.WHITE);
        }
        if (P < 0.5 && A > 0.5 && D < 0.5){
            //-+-
            changePanelColor("-+-", Color.GREEN);

        }else{
            changePanelColor("-+-", Color.WHITE);
        }

        if (P < 0.5 && A < 0.5 && D > 0.5){
            //--+
            changePanelColor("--+", Color.GREEN);

        }else{
            changePanelColor("--+", Color.WHITE);
        }

        if (P < 0.5 && A < 0.5 && D < 0.5){
            //---
            changePanelColor("---", Color.GREEN);

        }else{
            changePanelColor("---", Color.WHITE);
        }

    }




}
