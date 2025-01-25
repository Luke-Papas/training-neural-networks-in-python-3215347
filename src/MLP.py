import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) *2) -1
        #Number of Weights = # of inputs +1 for the bias
        #Add scaling factor of 2 and a shift of -1
        #Why?
        self.bias = bias
        #saving the bias term for later

    def run(self, x): # feeds the input array x -> into the perceptron to -> return the activation functions output
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights) # dot product A*B = a1b1+a2b2+a3b3
        return self.sigmoid(x_sum)
        