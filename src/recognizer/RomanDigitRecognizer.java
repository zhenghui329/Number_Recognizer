package recognizer;

import org.tensorflow.SavedModelBundle;

import java.awt.image.BufferedImage;

/*
 * ArabicDigitRecognizer is a class to recognize roman number.
 * It implements the TensorflowDigitRecognizer interface.
 *
 */
public class RomanDigitRecognizer implements TensorflowDigitRecognizer {
    private SavedModelBundle saved_model_bundle_;
    private boolean debug = true;

    RomanDigitRecognizer(){

    }

    /*
     * loadModel method loads a saved model from an export directory.
     * */
    @Override
    public void loadModel(String model_folder_path) {
        saved_model_bundle_ = SavedModelBundle.load(model_folder_path, "serve");
    }

    /*
     * recognize method takes a bufferedImage as input,
     * predicts and returns the number it represents.
     * */
    @Override
    public int recognize(BufferedImage img) {
        return 0;
    }
}
