'''
Created on 23-12-2011

@author: Freddie
'''
from layer import Layer
from utils import normalize, print_vector, der_sigmoid

class grossberg_layer(Layer):
    '''
    classdocs
    '''


    def __init__(self, type = "delta"):
        '''
        Constructor
        '''
        self.type = type
        self.neurons  = []
        
    def learn_func(self, arg):
        if(type == "widrow") :
            return 1
        return der_sigmoid(arg)
        
    def grossberg_learn(self, speed_factor, inputs, outputs,doutputs):
        inputs = normalize(inputs)
        #print_vector(inputs)
        for idx,neuron in enumerate(self.neurons):
            output = outputs[idx];
            doutput = doutputs[idx];
            #print output
            #print doutput
            #print_vector(neuron.weights)
            #print_vector(inputs)
            der = self.learn_func(neuron.wei_sum(inputs))
            #neuron.weights = [neuron.weights[i] + speed_factor*(doutput-output)*inputs[i]*der for i in range(len(inputs))] + [neuron.weights[len(inputs)] + speed_factor*(doutput-output)*(-1)*der]
            neuron.weights = [neuron.weights[i] + speed_factor*(doutput-output)*inputs[i]*der for i in range(len(inputs))] + [0.0]
            #neuron.weights = normalize(neuron.weights)
            #print idx
            #print_vector(neuron.weights)
            

    def find_winner_idx(self, inputs2):
        winner = 0
        widx = 0
        for idx,inp in enumerate(inputs2) :
            if(inp > winner) :
                winner = inp
                widx = idx
        return widx
    
    
    def grossberg_multilearn(self, images, doutputs, nn, epochs=10000):
        eta = 0.5;
        for epoch in range(1,epochs+1):
            if(epoch%2000==0):
                eta = eta /2
            for idx,doutput in enumerate(doutputs):
                outputs = nn.calculate(images[idx]);
                #print_vector(normalize(nn.layers[-2].outputs))
                #print_vector(self.outputs);
                #print_vector(doutput)
                #print_vector(outputs)
                inputs = []
                for idx,inp in enumerate(nn.layers[-2].outputs):
                    inputs.append(0);
                winner = self.find_winner_idx(nn.layers[-2].outputs)
                inputs[winner] = 1
                #print_vector(inputs)
                
                self.grossberg_learn(eta, inputs, outputs, doutput);
                
                
    def calculate_normal(self, inputs):
        outputs = []
        inps = []
        for idx,inp in enumerate(inputs):
            inps.append(0);
        winner = self.find_winner_idx(inputs)
        inps[winner] = 1
        for n in self.neurons:
            outputs.append(n.output(inps))
        self.outputs = outputs
        return outputs