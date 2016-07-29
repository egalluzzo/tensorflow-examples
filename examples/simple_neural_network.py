import math, random

def sigmoid(vec):
	return [1.0 / (1.0 + math.e ** -elem) for elem in vec]

def softmax(vec):
	exp = [math.e ** elem for elem in vec]
	total = sum(exp)
	return [elem / total for elem in exp]

class HiddenLayer:
	def __init__(self, n_inputs, n_outputs, activation):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.activation = activation
		self.weights = [[random.uniform(0, 1) for x in range(0, n_inputs)] for y in range(0, n_outputs)]
		self.bias = [random.uniform(0, 1) for y in range(0, n_outputs)]

	def apply(self, inputs):
		outputs = []
		for i in range(self.n_outputs):
			sum = 0
			for j in range(self.n_inputs):
				sum += self.weights[i][j] * inputs[j]
			sum += self.bias[i]
			outputs.append(sum)
		return self.activation(outputs)

inputs = [1, 2, 3]
hidden_layer = HiddenLayer(3, 4, sigmoid)
output_layer = HiddenLayer(4, 2, softmax)
outputs = output_layer.apply(hidden_layer.apply(inputs))
print("Input: {}".format(inputs))
print("Hidden layer:\n  Weights: {}\n  Bias: {}".format(hidden_layer.weights, hidden_layer.bias))
print("Output: {}".format(outputs))
