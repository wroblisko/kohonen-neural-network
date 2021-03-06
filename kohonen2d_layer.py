import math
from kohonen_layer import *
from distance_functions import *

class KohonenLayer2D(KohonenLayer):
    def __init__(self, fun=distance2D):
        self.neurons  = []
        self.neuron_position = {}
        self.position_counter = 0
        self.distance_function = fun
        self.neurons_in_row = 0

    def set_number_of_neurons(self, n):
        neurons_in_row = math.trunc(math.sqrt(n))
        if neurons_in_row!=math.sqrt(n):
            raise Exception, "Bad number of neurons in kohonen 2D layer"

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.neuron_position[neuron]=self.position_counter
        self.position_counter += 1
