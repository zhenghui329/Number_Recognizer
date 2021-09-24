package gui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.io.IOException;

/**
 * @web API doc: https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary
 * @web https://docs.oracle.com/javafx/2/get_started/fxml_tutorial.htm
 *
 */
public class Main extends Application {
    @Override
    public void start(Stage primaryStage) {
        try {
            Parent root = FXMLLoader.load(getClass().getResource("number_recognizer.fxml"));
            primaryStage.setTitle("Digit recognizer Application");
            primaryStage.setScene(new Scene(root, 800, 400));
            primaryStage.show();
        } catch(IOException ex){
            System.out.println("Error: FXMLLoader failed.");
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }

}
