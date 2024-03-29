import numpy as np
from library.activation import Activation
from library.layer import Layer
from library.utils import ReLU_function, ReLU_prime

# Used to Normalize data. Usually used as Output layer
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            s = s * (1 - s)
            return s

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    
    def __init__(self, use_cross_entropy: bool = False):
        Layer.__init__(self)
        self._use_cross_entropy = use_cross_entropy
        pass
    
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        if self._use_cross_entropy:
            return output_gradient
        else:    
            n = np.size(self.output)
            return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

class ReLU(Activation):
    def __init__(self):
        super().__init__(ReLU_function, ReLU_prime)