from utils import *

class Neuron:

    def __init__(self, weights, func=sigmoid, bias=True):
        self.weights = weights
        self.func = func
        self.der_func = eval('der_'+func.func_name)
        self.ro = 1
        self.bias = 1 if bias else 0

    def set_bias(self, bias):
        self.bias = 1 if bias else 0

    def output(self, args):
        self.inputs = args[:]                
        self.inputs.append(self.bias)
        self.y = self.function(self.wei_sum(self.inputs))
        return self.y
    
    def wei_sum(self, args):
        #bierze pod uwage bias (jest w args)
        total  = sum( [ self.weights[i] * args[i] for i in range(len(args)) ] )
        #total += self.bias * self.weights[-1]
        return total
    
    def function(self, arg):
        return self.func(arg)
      
    def set_function(self, func):
        self.func = func
