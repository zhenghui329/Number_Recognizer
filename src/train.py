import tensorflow as tf
import numpy as np
import spatial_transformer as spt
import shutil
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.examples.tutorials.mnist import input_data

# function for randomly generate initial weight variable with given shape
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# function for randomly generate initial bias variable with given shape
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# function for doing 2D convolution with input matrix x and weight W.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# function for doing 2x2 max pool operation on input matrix x.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# function for saving the model into output_folder.
def save_model(session, input_tensor, output_tensor, output_folder):
  print('save model to ' + output_folder)
  signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs = {'input': tf.saved_model.utils.build_tensor_info(input_tensor)},
    outputs = {'output': tf.saved_model.utils.build_tensor_info(output_tensor)},
  )
  b = saved_model_builder.SavedModelBuilder(output_folder)
  b.add_meta_graph_and_variables(session,
                                 [tf.saved_model.SERVING],
                                 signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
  b.save()


TRAINING_BATCH_NUM = 1000
TRAINING_BATCH_SIZE = 50
TRAIN_DATA_DIR = './MNIST_data/'
MODEL_OUTPUT_DIR = './model_' + str(TRAINING_BATCH_NUM) + '_batches_with_size_' + str(TRAINING_BATCH_SIZE) + '/'
if __name__=="__main__":
    # Remove existing data in output_dir.
    shutil.rmtree(MODEL_OUTPUT_DIR, True)
    # Read input dataset from MNIST_data folder.
    mnist = input_data.read_data_sets(TRAIN_DATA_DIR, one_hot=True)
    tf.reset_default_graph()
    # x is the placeholder for input image, y_ is the placeholder for input label.
    # keep_prob is the place holder for keep probabilities.
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # First layer of network (fully connected neural network), x is the input, h_fc_loc1_drop is the output.
    W_fc_loc1 = weight_variable([784, 20])
    b_fc_loc1 = bias_variable([20])
    # Linear transformation and apply tanh for non linearity.
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
    # Dropout operation.
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

    # Second layer of network (spatial transformer network), h_fc_loc1_drop is the input, h_trans is the output.
    W_fc_loc2 = weight_variable([20, 6])
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
    out_size = (28, 28)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_trans = spt.transformer(x_image, h_fc_loc2, out_size)

    # Third layer of network (convolutional neural netork), h_trans is the input, h_pool1 is the output.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 2D convolution and apply relu for non linearity.
    h_conv1 = tf.nn.relu(conv2d(h_trans, W_conv1) + b_conv1)
    # 2x2 max pool to shrink the output size.
    h_pool1 = max_pool_2x2(h_conv1)

    # Fourth layer of the network (convolutional neural netork), h_pool1 is the input, h_pool2 is the output.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 2D convolution and apply relu for non linearity.
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 2x2 max pool to shrink the output size.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fifth layer of the network (fully connected neural network), h_pool2 is the input, h_fc1_drop is the output.
    W_fc1 = weight_variable([7 * 7 * 64, 20])
    b_fc1 = bias_variable([20])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # Linear transformation and apply relu for non linearity.
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout operation.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer of the neural network. h_fc1_drop is the input, y_conv is the output.
    W_fc2 = weight_variable([20, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y_conv")

    print(y_conv)

    # Specify how the optimization is done.
    # y_ is the label in the training set, and y_conv is the output of the neural network.
    # Use cross entropy as the cost function and use softmax function to normalize the logits (the output y_conv).
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train with 50 input images per batch.
    # Save the model
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(TRAINING_BATCH_NUM):
        batch = mnist.train.next_batch(TRAINING_BATCH_SIZE)
        # Print the accuracy on training data every 100 batch for debugging purpose.
        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.7})
      save_model(sess, x, y_conv, MODEL_OUTPUT_DIR)
      # Print the prediction accuracy on mnist test data.
      print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))