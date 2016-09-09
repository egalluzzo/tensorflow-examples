import sys, os
sys.path.append(sys.path.append(os.path.dirname(__file__) + "/.."))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_neural_network import mnist_neural_network


def variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool_2x2(x, name):
	return tf.nn.max_pool(x,
		ksize = [1, 2, 2, 1],
		strides = [1, 2, 2, 1],
		padding = 'VALID',
		name = name)

def create_vars():
	keep_prob = tf.placeholder(tf.float32)
	return {"keep_prob": keep_prob}

def create_model(x, input_size, output_size, vars):
	conv_map_count_1 = 20
	conv_map_count_2 = 40
	hidden_layer_size = 800

	# Reshape the batch_size x 784 input into a batch_size x 28 x 28 input,
	# with a fourth dimension for number of channels (always 1 in our case
	# since it's a grayscale image).
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	# First convolutional layer, with conv_map_count_1 feature maps
	# Each feature map is 24 x 24, max pool => 12 x 12
	W_conv1 = variable([5, 5, 1, conv_map_count_1], "W_conv1")
	b_conv1 = variable([conv_map_count_1], "b_conv1")
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name = "h_conv1")
	h_pool1 = max_pool_2x2(h_conv1, "h_pool1")

	# Second convolutional layer, with conv_map_count_2 feature maps
	# Each feature map is 8 x 8, max pool => 4 x 4
	W_conv2 = variable([5, 5, conv_map_count_1, conv_map_count_2], "W_conv2")
	b_conv2 = variable([conv_map_count_2], "b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name = "h_conv2")
	h_pool2 = max_pool_2x2(h_conv2, "h_pool2")

	# Fully connected layer
	fc_input_size = 4 * 4 * conv_map_count_2
	W_fc = variable([fc_input_size, hidden_layer_size], "W_fc")
	b_fc = variable([hidden_layer_size], "b_fc")
	h_pool2_flat = tf.reshape(h_pool2, [-1, fc_input_size])
	h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc, name = "h_fc")

	# Apply dropout to the fully connected layer to reduce overfitting.
	h_fc_drop = tf.nn.dropout(h_fc, vars["keep_prob"])

	# Output layer
	W_out = variable([hidden_layer_size, output_size], "W_out")
	b_out = variable([output_size], "b_out")
	y = tf.nn.softmax(tf.matmul(h_fc_drop, W_out) + b_out, name = "y")

	return y

def create_var_values_for_train(vars):
	return {vars["keep_prob"]: 0.5}

def create_var_values_for_eval(vars):
	return {vars["keep_prob"]: 1.0}


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
nn = mnist_neural_network.MnistNeuralNetwork(
	mnist,
	create_model,
	create_vars,
	create_var_values_for_train,
	create_var_values_for_eval)
nn.train_and_evaluate()
