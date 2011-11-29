import math


def print_vector(vec):
    for num in vec:
        print '%.2f '%num,
    print ''

def sigmoid(total):
    return 1.0 / ( 1.0 + math.exp(- total) )

def linear(x):
    return x

def normalize(vec):
    n = math.sqrt(sum(map(lambda x: x*x, vec)))
    return map(lambda x: x/n, vec)

def substract(v1, v2):
    return [v1[it]-v2[it] for it in range(len(v1))]

