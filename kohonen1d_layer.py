from kohonen_layer import *
from distance_functions import *

class KohonenLayer1D(KohonenLayer):
    def __init__(self, fun=distance1D):
        self.neurons  = []
        self.neuron_position = {}
        self.position_counter = 0
        self.distance_function = fun

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.neuron_position[neuron]=self.position_counter
        self.position_counter += 1
