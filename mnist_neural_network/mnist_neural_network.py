import tensorflow as tf

class MnistNeuralNetwork:

	def __init__(self, mnist, create_model):
		# The function to create the model, given the input variable
		self.create_model = create_model

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
		self.y = self.create_model(self.x, self.input_size, self.output_size)

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
			if i % 100 == 0:
				train_accuracy = 100 * self.accuracy.eval(feed_dict = {self.x: batch[0], self.y_: batch[1]})
				print("Step {:05d}: batch training accuracy {:.0f}%".format(i, train_accuracy))
			train_step.run(feed_dict = {self.x: batch[0], self.y_: batch[1]})

	def evaluate(self):
		# Evaluate the network
		feed_dict = {self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}
		print("Accuracy: {:.3f}%".format(100 * self.accuracy.eval(feed_dict = feed_dict)))


	def train_and_evaluate(self):
		# Initialize TensorFlow session
		self.sess = tf.Session()
		with self.sess.as_default():
			self.train()
			self.evaluate()
