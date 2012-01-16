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
            outputs = layer.calculate_normal(inputs)
            inputs = outputs #for next iteration
        return self.layers[-1].outputs

    """to samo ale bez normalizacji"""
    def calculate(self, inputs):
        inputs = normalize(inputs)
        return self.calculate_normal(inputs)
    
    def describe(self):
        for (i,layer) in enumerate(self.layers):
            print "Layer ",i
            layer.print_weights()   
                
    def set_bias(self, bias):
        for l in self.layers:
            l.set_bias(bias)                

    def bp_learn_step(self, inputs, outputs, eta=0.5, momentum=0.1):
        debug = False     
        if debug:
            print "Learing ",inputs
        y = self.calculate_normal(inputs)
        if debug:
            print "Output is ",y
        der_func = self.layers[-1].neurons[0].der_func        
        errors = [(i-j)*der_func(j) for (i,j) in zip(outputs,y)]
        if debug:
            print "Total errors= ",errors        
        for (i,errors) in enumerate(errors):
            self.layers[-1].neurons[i].error = errors     
        #calculating errors - back
        layers_to_calculate = list(enumerate(self.layers[:-1]))
        layers_to_calculate.reverse()
        for (i,layer) in layers_to_calculate:        
            next_layer = self.layers[i+1]
            for (neuron_i, neuron) in enumerate(layer.neurons):
                neuron.error = sum([next_layer_neuron.error*next_layer_neuron.weights[neuron_i] for next_layer_neuron in next_layer.neurons])
                neuron.error = neuron.error*neuron.der_func(neuron.y)
                if debug:
                    print "Neuron ",neuron_i," in layer ",i," error= ", neuron.error
        #calculating modified weights
        if debug:
            print "Modyfing weights...."
        for layer in self.layers:
            for neuron in layer.neurons:
                e = neuron.wei_sum(neuron.inputs) #inputs zawiera wejscie bias
                der_func_value = neuron.der_func(e)
                neuron.weights = [neuron.weights[i] + 
                                  eta*neuron.error*der_func_value*neuron.inputs[i] + 
                                  momentum*neuron.last_changes_weights[i] for i in range(len(neuron.weights))]
                neuron.last_changes_weights = [neuron.error*der_func_value*neuron.inputs[i] for i in range(len(neuron.weights))]
        if debug:
            self.describe()
            print "\n"
                
    def bp_learn(self, inputs, outputs, epochs=1000, eta=0.5, momentum=0.1):
        #reseting momentum's inits
        for l in self.layers:
            for n in l.neurons:
                n.last_changes_weights = [0]*len(n.weights)
        input_i = 0
        for _ in range(epochs):
            input_to_learn = inputs[input_i]
            output_to_learn = outputs[input_i]            
            input_i = (input_i + 1)%len(inputs)
            self.bp_learn_step(input_to_learn, output_to_learn, eta, momentum)
            
    def print_error(self, inputs, outputs):
        y = self.calculate_normal(inputs)        
        errors = [i-j for (i,j) in zip(outputs,y)]
        print "Total Errors = ",errors