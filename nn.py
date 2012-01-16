#!/usr/bin/python

import sys
import math
import random
from distance_functions import *
from neuron import *
from utils import *
from neural_network import *        
                        


bp_images = [[1, 0, 1, 1, 0, 1, 1, 0, 1],
             [1, 1, 1, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 1, 0, 0, 0, 0]]

NN = NeuralNetwork("mlp_xor.txt")

print "================================"
print "Input images:"
for idx,image in enumerate(bp_images):
    print "Image",idx,"=",
    print_vector(image)
NN.set_bias(True)
NN.describe()
print "================================\n\n"
#NN.bp_learn([0, 1], [0, 1])
#NN.bp_learn([1, 0], [1, 0])

debug = True
#NN.bp_learn([[0,0],[1,0],[0,1],[1,1]], [[0],[1],[1],[0]], epochs=5000, eta=0.5, momentum=0.1)
#
print "================================\n\n"
#NN.print_error([0, 0], [0])
#NN.print_error([0, 1], [1])
#NN.print_error([1, 0], [1])
#NN.print_error([1, 1], [0])
#NN.print_error([0, 1], [0, 1])
#NN.print_error([1, 0], [1, 0])

def get_in(set):
    return [i for i,o in set]
def get_out(set):
    return [o for i,o in set]


parity = [
          [[0,0,1],[0]],
          [[0,1,0],[0]],
          [[1,0,0],[0]],
          [[0,1,1],[1]],
          [[1,0,1],[1]],
          [[1,1,0],[1]],
          [[1,1,1],[0]],
          [[0,0,0],[0]],
           ]

xor = [
       [[0,0],[0]],
       [[0,1],[1]],
       [[1,0],[1]],
       [[1,1],[0]],
       ]
#NN.bp_learn(get_in(parity), get_out(parity), epochs=10000, eta=0.1, momentum=0.0)
NN.bp_learn(get_in(xor), get_out(xor), epochs=10000, eta=0.5, momentum=0.2)

for i,o in xor:
    NN.print_error(i,o)

exit()











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
