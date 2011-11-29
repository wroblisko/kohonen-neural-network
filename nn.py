#!/usr/bin/python

import sys
import math
import random


def sigmoid(total):
    return 1.0 / ( 1.0 + math.exp(- total) )

def linear(x):
    return x

def normalize(vec):
    n = math.sqrt(sum(map(lambda x: x*x, vec)))
    return map(lambda x: x/n, vec)

def substract(v1, v2):
    return [v1[it]-v2[it] for it in range(len(v1))]

class Neuron:

    def __init__(self, weights, func=sigmoid, ro=0.75):
        self.weights = weights
        self.func = func
        self.ro = ro

    def output(self, args):
        # bias self.weights[-1]
        total  = sum( [ self.weights[i] * args[i] for i in range(len(args)) ] )
        total += (-1) * self.weights[-1]
        return self.function(total)
    
    def function(self, arg):
        return self.func(arg)
      
    def set_function(self, func):
        self.func = func


class Layer:
    def __init__(self):
        self.neurons  = []
		
    def add_neuron(self, neuron):
        self.neurons.append(neuron)


class KohonenLayer(Layer):
    def __init__(self):
        self.neurons  = []
        self.neuron_position = {}
        self.position_counter = 0

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        self.neuron_position[neuron]=self.position_counter
        self.position_counter += 1
            
        
class NeuralNetwork:

    def __init__(self, argv):
		self.layers = []
                self.ro_min = 0.75

		f = open(argv[0], 'r')

		[networkInputs, layersNum] = [int(x) for x in f.readline().split()]
		neuronsNums = [networkInputs]
		for i in range(layersNum):
			layerDescription = f.readline().split()
			neuronsNums.append(int(layerDescription[0]))
                        layerType = layerDescription[1]
			activationFun = layerDescription[2]
			
                        if layerType=="normal":
                            L = Layer()	
                        else:
                            L = KohonenLayer()
                        L.num_inputs = neuronsNums[-2]
			
			for j in range(neuronsNums[-1]):
				if len(layerDescription) == 5:
					weights = [ random.uniform(float(layerDescription[3]), float(layerDescription[4])) for i in (range(neuronsNums[-2]+1)) ]
				else:
					weights = [float(x) for x in f.readline().split()]
                                if layerType=="kohonen":
                                    weights[-1] = 0.0
                                weights = normalize(weights)
                                L.add_neuron(Neuron(weights, globals()[activationFun]))

			self.add_layer(L)
		f.close()

    def add_layer(self, layer):
        self.layers.append(layer)
		
    def show(self):
        print "Inputs:", self.layers[0].num_inputs
        for idx, layer in enumerate(self.layers):
            print "Layer %d: %d neurons outval=" % (idx, len(layer.neurons)), layer.output   
            for idx, neuron in enumerate(layer.neurons):
                print "Neuron %d: weights="%(idx,),neuron.weights
        print "Outputs:", len(self.layers[-1].neurons)
        
    def calculate(self, inputs):
        inputs = normalize(inputs)
        #if len(inputs) != len(self.layers[0].neurons):
        #    raise ValueError, 'wrong number of inputs'
        
        for lid, layer in enumerate(self.layers):
            outputs = []
            for n in layer.neurons:
                outputs.append( n.output(inputs) )
            inputs = outputs #for next iteration
            layer.output =  outputs
    
        return self.layers[-1].output
    
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
        winner_position = self.layers[-1].neuron_position[winner]
        winner.weights = [winner.weights[i] + speed_factor*(inputs[i]-winner.weights[i]) for i in range(len(inputs))] + [0.0]
        
        

def print_vector(vec):
    for num in vec:
        print '%.2f '%num,
    print ''
                

NN = NeuralNetwork(["kohonen.txt"])
#print NN.test([1,2,3,4,5,6,7,8,9])
#print NN.calculate([3,6])
NN.calculate([2,1])
NN.show()
#print "[2,1] normalize=",normalize([2,1])
images = [ [1, 1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 1, 0, 0, 0]]

for i in range(8000):
    print "==========================================="
    for image in images:
        NN.kohonen_learn(0.1, image)
    for (idx, image) in enumerate(images):
        NN.calculate(image)
        print "img",idx+1,"-> ",
        print_vector(NN.layers[-1].output)

#NN.kohonen_learn(0.1, [2,1])
#NN.show() 
#print NN.test(
#[float(x) for x in sys.argv[2:]])