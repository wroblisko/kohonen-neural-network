#!/usr/bin/python

import sys
import math
import random
from distance_functions import *
from neuron import *
from utils import *
from neural_network import *        
                        

NN = NeuralNetwork("parity.txt")

NN.layers[-2].distance_function = empty_distance
NN.layers[-2].ro_min = 0.8

NN.layers[-1].print_weights()

images = [ [1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 1, 0, 1, 0, 0]
           ]

images = readInputFile("input.in")

doutputs = [  [1,0,0],[1,0,0],[1,0,0],
              [0,1,0],[0,1,0],[0,1,0],
              [0,0,1],[0,0,1],[0,0,1]
            ]
doutputs = readInputFile("outputs.in")
images = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
doutputs = [[0],[1],[1],[0],[1],[0],[0],[1]]
print "================================"
print "Input images:"
for idx,image in enumerate(images):
    print "Image",idx,"=",
    print_vector(image)
print "================================"
print "Normalized input images:"
for idx,image in enumerate(images):
    print "Image",idx,"=",
    print_vector(normalize(image))

print "=============================="
print "Learning..."

NN.layers[-2].kohonen_multilearn(images)

for (idx, image) in enumerate(images):
    print "Image",idx+1,"-> ",
    NN.calculate(image)
    print_vector(NN.layers[-2].outputs)

NN.layers[-1].grossberg_multilearn(images,doutputs,NN)


NN.layers[-2].print_weights(columns=9, bias=True)
NN.layers[-1].print_weights(columns=9, bias=True)
print "Testing for images.."
for (idx, image) in enumerate(images):
    print "Image",idx+1,"-> ",
    print_vector(NN.calculate(image))






exit()
for i in range(8000):
    print "==========================================="
    for image in images:
    #image = images[random.randint(0,3)]
        NN.kohonen_learn(0.1, image)
    for (idx, image) in enumerate(images):
        print "Image",idx+1,"-> ",
        print_vector(NN.calculate(image))
