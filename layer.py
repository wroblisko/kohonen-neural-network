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

    """zwraca i ustawia pole outputs w warstwie"""
    def calculate_normal(self, inputs):
        outputs = []
        for n in self.neurons:
            outputs.append(n.output(inputs))
        self.outputs = outputs
        return outputs

    def calculate(self, inputs):
        return self.calculate_normal(normalize(inputs))
