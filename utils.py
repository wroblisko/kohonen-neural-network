import math


def print_vector(vec, break_line=None):
    for idx,num in enumerate(vec):
        if break_line is not None and idx%break_line==0:
            print ""
        print '%.5f '%num,
    print ''

def sigmoid(total):
    return 1.0 / ( 1.0 + math.exp(- total) )

def der_sigmoid(total):
    return sigmoid(total) * (1-sigmoid(total))

def tanh(total):
    return math.tanh(total)

def der_tanh(total):
    return 1-math.pow(math.tanh(total),2)

def linear(x):
    return x

def der_linear(x):
    return 1

def normalize(vec):
    if sum(vec)==0:
        return vec
    n = math.sqrt(sum(map(lambda x: x*x, vec)))
    return map(lambda x: x/n, vec)

def substract(v1, v2):
    return [v1[it]-v2[it] for it in range(len(v1))]

def readInputFile(name):
    outVec = []
    for line in open(name, 'r').readlines():
        vec = []
        for token in line.split() :
            vec.append(float(token))
        outVec.append(vec)
    return outVec
    

