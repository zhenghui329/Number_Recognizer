package recognizer;

import java.awt.image.BufferedImage;


/*
 * TensorflowDigitRecognizer is a interface for
 * recognizing digits in an image based on tensorflow.
 */

public interface TensorflowDigitRecognizer {
    // Loads the tensorflow model given the model folder path.
    void loadModel(String model_folder_path);

    // Takes in an image and returns an int representing the digit it belongs to.
    int recognize(BufferedImage img);
}