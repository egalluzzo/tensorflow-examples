import tensorflow as tf

class MnistNeuralNetwork:

	def __init__(self, mnist, create_model, create_vars = None, create_var_values_for_train = None, create_var_values_for_eval = None):
		# The functions to create the model, any additional variables, and their values
		self.create_model = create_model
		self.create_vars = create_vars
		self.create_var_values_for_train = create_var_values_for_train
		self.create_var_values_for_eval = create_var_values_for_eval

		# Hyperparameters
		self.batch_size = 50
		self.learning_rate = 0.05
		self.iterations = 4000

		# Other constants
		self.input_size = 28 * 28
		self.output_size = 10

		# Read data
		self.mnist = mnist


	def train(self):
		# Inputs
		self.x = tf.placeholder(tf.float32, shape = [None, self.input_size], name = "x")
		self.y_ = tf.placeholder(tf.float32, shape = [None, self.output_size], name = "y_")

		# Output
		if (self.create_vars != None):
			self.vars = self.create_vars()
			self.y = self.create_model(self.x, self.input_size, self.output_size, self.vars)
			var_values_for_train = self.create_var_values_for_train(self.vars)
		else:
			self.vars = None
			self.y = self.create_model(self.x, self.input_size, self.output_size)
			var_values_for_train = {}

		# Cost
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices = [1]), name = "cross_entropy")

		# Accuracy
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy")

		# Initialize variables
		self.sess.run(tf.initialize_all_variables())

		# Training
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
		for i in range(self.iterations):
			batch = self.mnist.train.next_batch(self.batch_size)
			standard_feed_dict = {self.x: batch[0], self.y_: batch[1]}
			feed_dict = dict(standard_feed_dict.items() + var_values_for_train.items())
			if i % 100 == 0:
				train_accuracy = 100 * self.accuracy.eval(feed_dict = feed_dict)
				print("Step {:05d}: batch training accuracy {:.0f}%".format(i, train_accuracy))
			train_step.run(feed_dict = feed_dict)

	def evaluate(self):
		# Evaluate the network
		if self.create_var_values_for_eval != None:
			var_values_for_eval = self.create_var_values_for_eval(self.vars)
		else:
			var_values_for_eval = {}

		standard_feed_dict = {self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}
		feed_dict = dict(standard_feed_dict.items() + var_values_for_eval.items())
		print("Accuracy: {:.3f}%".format(100 * self.accuracy.eval(feed_dict = feed_dict)))


	def train_and_evaluate(self):
		# Initialize TensorFlow session
		self.sess = tf.Session()
		with self.sess.as_default():
			self.train()
			self.evaluate()
