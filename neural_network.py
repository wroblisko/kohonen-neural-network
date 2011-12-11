import random
from utils import *
from neuron import *
from layer import *
from kohonen_layer import *
from kohonen2d_layer import *
        
class NeuralNetwork:

    def __init__(self, argv, ro_min = 0.5, kohonen1D_fun=distance1D, kohonen2D_fun=distance2D):
		self.layers = []
                self.ro_min = ro_min

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
                            L = KohonenLayer(fun=kohonen1D_fun)
                        elif layerType=="kohonen2D":
                            L = KohonenLayer2D(fun=kohonen2D_fun)
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
                                weights = normalize(weights)

                            #dla 2D musimy znac wielkosc siatki
                            if layerType=="kohonen2D":                                
                                L.set_number_of_neurons(neurons_number[-1])

                            L.add_neuron(Neuron(weights, eval(activation_function)))

			self.add_layer(L)
		f.close()

    def add_layer(self, layer):
        self.layers.append(layer)
		
    """oblicza bez normalizacji"""
    def calculate_normal(self, inputs):
        for lid, layer in enumerate(self.layers):
            outputs = []
            for n in layer.neurons:
                outputs.append( n.output(inputs) )
            inputs = outputs #for next iteration
            layer.output =  outputs
    
        return self.layers[-1].output

    """normalizuje wektor i oblicza"""
    def calculate(self, inputs):
        inputs = normalize(inputs)
        return self.calculate_normal(inputs)
    
    """znajduje neuron ktory daje najwyzsze wyjscie w ostatniej warstwie (narazie jest to warstwa kohonena)
    zwraca neuron ktory wygral i nie jest zmeczony"""
    def find_winner(self, inputs):
        outputs = self.calculate(inputs)
        #znajdowanie max. wyjscia i niezmeczonego neurona
        i_max, val_max = 0, 0
        for (i, val) in enumerate(outputs):
            neuron = self.layers[-1].neurons[i]
            if val>val_max and neuron.ro >= self.ro_min:
                val_max, i_max= val, i
        #zmeczenie zwyciezcy i odpoczynek reszty
        n = len(self.layers[-1].neurons)
        for (idx, neuron) in enumerate(self.layers[-1].neurons):
                neuron.ro = neuron.ro-self.ro_min if idx==i_max else neuron.ro+1.0/n
                neuron.ro = min(neuron.ro, 1)
        return self.layers[-1].neurons[i_max]

    def kohonen_learn(self, speed_factor, inputs):
        winner = self.find_winner(inputs)
        inputs = normalize(inputs)
        layer = self.layers[-1]
        #obsluga sasiedztwa wielowymiarowego
        for neuron in layer.neurons:
            h = layer.distance(neuron, winner)
            neuron.weights = [neuron.weights[i] + h*speed_factor*(inputs[i]-neuron.weights[i]) for i in range(len(inputs))] + [0.0]
            neuron.weights = normalize(neuron.weights)
            
    def kohonen_multilearn(self, images, epochs=8000, speed_factor=0.1, alfa=0.0001):
        for epoch in range(1,epochs+1):
            eta = max(speed_factor - alfa*epoch, 0)
            for image in images:
                self.kohonen_learn(eta, image)
            
