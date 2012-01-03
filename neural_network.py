import random
from utils import *
from neuron import *
from layer import *
from kohonen1d_layer import *
from kohonen2d_layer import *
from GrossbergLayer import grossberg_layer
        
class NeuralNetwork:

    def __init__(self, argv, ro_min = 0.5):
		self.layers = []

		f = open(argv, 'r')

		[network_inputs, layersNum] = [int(x) for x in f.readline().split()]
		neurons_number = [network_inputs]
		for i in range(layersNum):
			layer_description = f.readline().split()
			neurons_number.append(int(layer_description[0]))
                        layerType = layer_description[1]
			activation_function = layer_description[2]
			
                        if layerType=="normal":
                            L = Layer()	
                        elif layerType=="kohonen":
                            L = KohonenLayer1D()
                        elif layerType=="kohonen2D":
                            L = KohonenLayer2D()
                        elif layerType=="grossberg":
                            L = grossberg_layer()
                        else:
                            raise Exception, "Bad layer"

                        L.num_inputs = neurons_number[-2]
			
                        #dla kazdego neuronu z wczytanej warstwy losuj badz wczytaj wagi
			for j in range(neurons_number[-1]):                                
                            if len(layer_description) == 5:
                                weights = [ random.uniform(float(layer_description[3]), float(layer_description[4])) for i in (range(neurons_number[-2]+1)) ]
                            else:
                                weights = [float(x) for x in f.readline().split()]

                            #normalizacja i zerowanie biasu tylko dla kohonena
                            if layerType=="kohonen" or layerType=="kohonen2D":
                                weights[-1] = 0.0
                                #weights = normalize(weights)

                            #dla 2D musimy znac wielkosc siatki
                            if layerType=="kohonen2D":                                
                                L.set_number_of_neurons(neurons_number[-1])

                            L.add_neuron(Neuron(weights, eval(activation_function)))

			self.add_layer(L)
		f.close()

    def add_layer(self, layer):
        self.layers.append(layer)

    """zwraca wartosci neuronow wyjsciowych bez normalizacji"""
    def calculate_normal(self, inputs):
        for lid, layer in enumerate(self.layers):
            outputs = layer.calculate(inputs)
            inputs = outputs #for next iteration
        return self.layers[-1].outputs

    """to samo ale bez normalizacji"""
    def calculate(self, inputs):
        inputs = normalize(inputs)
        return self.calculate_normal(inputs)
            
