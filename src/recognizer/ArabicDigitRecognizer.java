package recognizer;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.Arrays;

/*
 * ArabicDigitRecognizer is a class to recognize arabic number from 0 ~ 9.
 * It implements the TensorflowDigitRecognizer interface.
 *
 */
public class ArabicDigitRecognizer implements TensorflowDigitRecognizer {
    private SavedModelBundle saved_model_bundle_;
    private boolean debug = true;

    ArabicDigitRecognizer(){
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
        // Get the input image width and height.
        int width = img.getWidth();
        int height = img.getHeight();

        // Create input buffer from the image.
        FloatBuffer fb = getFloatBuffer(img,width,height);

        // Create the keep probability array.
        float[] keep_prob_arr = new float[20];
        Arrays.fill(keep_prob_arr, 1f);

        // Get the Session with which to perform computation using the model
        Session tf_session = saved_model_bundle_.session();

        // Create tensor with data from the given buffer for input and keep_prob,
        // and run tensorflow prediction.
        try(Tensor inputTensor = Tensor.create(new long[] {1, width * height}, fb);
            Tensor keep_prob = Tensor.create(new long[] {1, 20}, FloatBuffer.wrap(keep_prob_arr));
            // runner(): create a Runner to execute graph operations and evaluate Tensors
            // feed(): feed the runner with the tensor
            // fetch(): fetch the output computed by the model, make run() return the output
            // run(): execute the graph fragments necessary to compute all requested fetches,
            //        return a list of Tensors
            Tensor result_tensor = tf_session.runner()
                    .feed("x", inputTensor)
                    .feed("keep_prob", keep_prob)
                    .fetch("y_conv")
                    .run()
                    .get(0))
        {
            // Get the predicted number from the result tensor
            int predict = getPredictedNum(result_tensor);
            System.out.println("prediction done");
            System.out.println(predict);
            return predict;
        }
    }

    /*
    *  getFloatBuffer method creates input buffer from the image pixel values.
    * */
    private FloatBuffer getFloatBuffer(BufferedImage img, int width,int height){
        FloatBuffer fb = FloatBuffer.allocate(width * height);
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                // (var & 0xFF): mask the variable so it leaves only the value in the last 8 bits
                // / 255.0f: only get the blue color from RGB, and convert it to a value between 0 and 1
                fb.put((float)(img.getRGB(col, row) & 0xFF)/255.0f);
                if(debug){
                    System.out.print((float)(img.getRGB(col, row) & 0xFF)/255.0f);
                    System.out.print(" ");
                }
            }
            if(debug) {
                System.out.print("\n");
            }
        }
        fb.rewind();
        return fb;
    }

    /*
    * getPredictedNum method gets the predicted number
    * from the result tensor.
    * It converts the tensor to an array and get the predicted digit by
    * checking the largest value in the result array,
    * then returns the index of the max value.
    * */
    private int getPredictedNum(Tensor result_tensor){
        float[][] result_array = new float[1][10];
        result_tensor.copyTo(result_array);
        float maxVal = Integer.MIN_VALUE;
        int inc = 0;
        int predict = -1;
        for(float val : result_array[0]) {
            if(debug){
                System.out.println(val);
            }
            if(val > maxVal) {
                predict = inc;
                maxVal = val;
            }
            inc++;
        }
        return predict;
    }
}
