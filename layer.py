from utils import *

class Layer:
    def __init__(self):
        self.neurons  = []
		
    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    """wyswietla wagi warstwy, moze dzielic na kolumny oraz nie drukowac biasu"""
    def print_weights(self, columns=None, bias=True):
        print "Layer : %d neurons" % len(self.neurons)
        for idx, neuron in enumerate(self.neurons):
            print "Neuron %d: weights="%(idx,),
            if bias:
                weights = neuron.weights[:]
            else:
                weights = neuron.weights[:-1]
            print_vector(weights, columns)
