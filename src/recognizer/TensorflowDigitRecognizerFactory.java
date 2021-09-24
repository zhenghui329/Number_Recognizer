package recognizer;


/*
 * TensorflowDigitRecognizerFactory is a factory class
 * to generate different type of recognizer
 * corresponding to the parameter
 */

public class TensorflowDigitRecognizerFactory {
    public TensorflowDigitRecognizer getDigitRecognizer(String recognizerType){
        if(recognizerType == null){
            return null;
        }
        if(recognizerType.equalsIgnoreCase("Arabic")) {
            ArabicDigitRecognizer rg = new ArabicDigitRecognizer();
            rg.loadModel("./src/model_100000_batches_with_size_50");
            return rg;
        }
        if(recognizerType.equalsIgnoreCase("Roman")){
            return new RomanDigitRecognizer();
        }
        return null;
    }
}