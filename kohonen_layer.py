from layer import *
from distance_functions import *

class KohonenLayer(Layer):
    def __init__(self, fun=distance1D):
        self.neurons  = []
        self.neuron_position = {}
        self.position_counter = 0
        self.distance_function = fun

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.neuron_position[neuron]=self.position_counter
        self.position_counter += 1

    def distance(self, n1, n2):
        return self.distance_function(self, n1, n2)
