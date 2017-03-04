# importing dependencies
from numpy import array,dot,random,exp
import numpy as np
# defining class
class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self):
		# seting seed so it can generate same random no. on every go
		random.seed(1)
		# defining synaptic weights randomly
		self.syn0 = 2*random.random((3,4))-1
		self.syn1 = 2*random.random((4,3))-1
	# defining sigmoid function
	def sigmoid(self,x,deriv = False):
		if deriv == True:
			return x*(1-x)
		return 1/(1+exp(-x))
	# defining train function 
	def train(self,trainingInputs,trainingOutputs,iterations):
		# iterations for  trainining
		for j in xrange(iterations):
			# feed forwarding to get l1 and l2
			l0,l1,l2 = self.think(trainingInputs)
			# determinig error on last layer
			l2_error = trainingOutputs-l2
			if (j%10000)==0:
				print "Error"+str(np.mean(np.abs(l2_error)))
			# finding l2 delta  for backprpogating of error
			l2_delta = l2_error*self.sigmoid(l2,deriv=True)
			# determining error on l1
			l1_error = l2_delta.dot(self.syn1.T)
			# finding l1 delta for back propogation
			l1_delta = l1_error*self.sigmoid(l1,deriv=True)
			# adjusting synaptic weights based on errors 
			self.syn1 += l1.T.dot(l2_delta)
			self.syn0 += l0.T.dot(l1_delta)
	# function for feedforwarding
	def think(self,inputs):
		l0 = inputs
		l1 = self.sigmoid(dot(l0,self.syn0))
		l2 = self.sigmoid(dot(l1,self.syn1))
		return l0,l1,l2 

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.syn0
    print neural_network.syn1

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 100000)

    print "New synaptic weights after training: "
    print neural_network.syn0
    print neural_network.syn1

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    l0,l1,l2= neural_network.think(array([0, 0, 0]))
    print l2
