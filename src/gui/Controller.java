package gui;

import recognizer.TensorflowDigitRecognizer;
import recognizer.TensorflowDigitRecognizerFactory;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.text.Text;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Controller{
    private final int CANVAS_WIDTH = 400;
    private final int CANVAS_HEIGHT = 400;
    public Canvas canvas = new Canvas(CANVAS_WIDTH,CANVAS_HEIGHT);
    private GraphicsContext gc;
    public ChoiceBox<String> modeChoiceBox;
    private String mode = "Arabic";
    public Line line = new Line(400,300,800,300);
    public Button recog_bt;
    public Button clear_bt;
    public Text num_label;

    public void initialize() {
        initCanvas();
        addListener_ChoiceBox();
        addEvent_Canvas();
    }


    // Initialize the Canvas
    private void initCanvas() {
        gc = canvas.getGraphicsContext2D();
        // Set canvas color to black
        gc.setFill(Color.BLACK);
        // Draw rectangle
        gc.fillRect(0,        // x of the upper left corner
                0,           // y of the upper left corner
                CANVAS_WIDTH,    // width of the rectangle
                CANVAS_HEIGHT);  // height of the rectangle
        gc.setStroke(Color.WHITE);
        gc.setLineWidth(10);
        gc.setLineCap(StrokeLineCap.SQUARE);
    }

    // Add listener to the mode choice box
    private void addListener_ChoiceBox(){
        // Add listener to modeChoiceBox
        modeChoiceBox.getSelectionModel().selectedItemProperty().addListener(new ChangeListener<String>() {
            // if the item of the list is changed
            public void changed(ObservableValue<? extends String> selected, String oldValue, String newValue) {
                // set the text for the mode to the selected item
                // and clear the canvas
                mode = newValue;
                gc.setFill(Color.BLACK);
                gc.fillRect(0,
                        0,
                        CANVAS_WIDTH,
                        CANVAS_HEIGHT);
                num_label.setText("");
            }
        });
    }

    // Add event to canvas to get the input img
    private void addEvent_Canvas(){
        canvas.addEventHandler(MouseEvent.MOUSE_PRESSED,
                new EventHandler<MouseEvent>(){
                    @Override
                    public void handle(MouseEvent event) {
                        System.out.println("mouse pressed");
                        gc.beginPath();
                        gc.moveTo(event.getX(), event.getY());
                        gc.stroke();
                    }
                });

        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED,
                new EventHandler<MouseEvent>(){
                    @Override
                    public void handle(MouseEvent event) {
                        gc.lineTo(event.getX(), event.getY());
                        gc.stroke();
                    }
                });

        canvas.addEventHandler(MouseEvent.MOUSE_RELEASED,
                new EventHandler<MouseEvent>(){
                    @Override
                    public void handle(MouseEvent event) {
                        System.out.println("mouse release");
                    }
                });
    }

    // method for button recog_bt
    public void recognizeNumber(ActionEvent actionEvent){
        BufferedImage scaledImg = getScaledImage(canvas);
        // Use a recognizerFactory to create recognizer by mode
        TensorflowDigitRecognizerFactory fc = new TensorflowDigitRecognizerFactory();
        TensorflowDigitRecognizer recognizer = fc.getDigitRecognizer(mode);
        // Recognize the input and set the result
        num_label.setText(Integer.toString(recognizer.recognize(scaledImg)));
    }

    // method for button clear_bt
    public void clear(ActionEvent actionEvent){
        gc.setFill(Color.BLACK);
        gc.fillRect(0,
                0,
                CANVAS_WIDTH,
                CANVAS_HEIGHT);
        num_label.setText("");
    }

    // Shrink the input img(400*400) to 28 * 28
    private BufferedImage getScaledImage(Canvas canvas) {
        // for a better recognition we should improve this part of how we retrieve the image from the canvas
        WritableImage writableImage = new WritableImage(CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.snapshot(null, writableImage);
        Image tmp =  SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
        BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = scaledImg.getGraphics();
        graphics.drawImage(tmp, 0, 0, null);
        graphics.dispose();
        return scaledImg;
    }


}
