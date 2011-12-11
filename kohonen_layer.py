from layer import *
from distance_functions import *

class KohonenLayer(Layer):
    def distance(self, n1, n2):
        return self.distance_function(self, n1, n2)
   
    """znajduje neuron ktory daje najwyzsze wyjscie w ostatniej warstwie (narazie jest to warstwa kohonena)
    zwraca neuron ktory wygral i nie jest zmeczony"""
    def find_winner(self, inputs):
        outputs = self.calculate(inputs)
        #znajdowanie max. wyjscia i niezmeczonego neurona
        i_max, val_max = 0, 0
        for (i, val) in enumerate(outputs):
            neuron = self.neurons[i]
            if val>val_max and neuron.ro >= self.ro_min:
                val_max, i_max= val, i
        #zmeczenie zwyciezcy i odpoczynek reszty
        n = len(self.neurons)
        for (idx, neuron) in enumerate(self.neurons):
                neuron.ro = neuron.ro-self.ro_min if idx==i_max else neuron.ro+1.0/n
                neuron.ro = min(neuron.ro, 1)
        return self.neurons[i_max]

    def kohonen_learn(self, speed_factor, inputs):
        winner = self.find_winner(inputs)
        inputs = normalize(inputs)
        #obsluga sasiedztwa wielowymiarowego
        for neuron in self.neurons:
            h = self.distance(neuron, winner)
            neuron.weights = [neuron.weights[i] + h*speed_factor*(inputs[i]-neuron.weights[i]) for i in range(len(inputs))] + [0.0]
            neuron.weights = normalize(neuron.weights)
            
    def kohonen_multilearn(self, images, epochs=8000, speed_factor=0.1, alfa=0.0001):
        for epoch in range(1,epochs+1):
            eta = max(speed_factor - alfa*epoch, 0)
            for image in images:
                self.kohonen_learn(eta, image)
