from utils import *

class Neuron:

    def __init__(self, weights, func=sigmoid):
        self.weights = weights
        self.func = func
        self.ro = 1

    def output(self, args):
        # bias self.weights[-1]
        total  = sum( [ self.weights[i] * args[i] for i in range(len(args)) ] )
        total += (-1) * self.weights[-1]
        return self.function(total)
    
    def function(self, arg):
        return self.func(arg)
      
    def set_function(self, func):
        self.func = func
