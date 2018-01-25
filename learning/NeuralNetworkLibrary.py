from random import *
import numpy as np
from math import *

class Neurone():
    def __init__(self,n,network):
        self.network = network
        self.threshold = 2*random()-1
        self.weights = [2*random()-1 for i in range(n)]
        self.n = n
        self.last_s = 0
        self.last_output = 0
    def output(self,input):
        s = self.threshold
        for i in range(self.n):
            s += self.weights[i]*input[i]
        self.last_s = s
        res = 1/(1+exp(-self.network.LAMBDA*s))
        self.last_output = res
        return(res)
    
class Layer():
    def __init__(self,ni,no,id,network):
        self.network = network
        self.ni = ni
        self.no = no
        self.neurones = [Neurone(ni,network) for i in range(no)]
        self.id = id
        self.error = None
        
    def output(self,input):
        return([neurone.output(input) for neurone in self.neurones])
    def retropropagate(self):
        try:
            errors = self.network.layers[self.id+1].error
        except:
            errors = [self.network.expected_output[i]-self.network.last_output[i] for i in range(self.no)]
            self.error = []
            for i in range(self.no):
                neurone = self.neurones[i]
                s = neurone.last_s
                derivate = self.network.LAMBDA*exp(-self.network.LAMBDA*s)/((1+exp(-self.network.LAMBDA*s))**2)
                self.error.append(derivate*errors[i])
            return()
        self.error = []
        for i in range(self.no):
            neurone = self.neurones[i]
            s = neurone.last_s
            derivate = self.network.LAMBDA*exp(-self.network.LAMBDA*s)/((1+exp(-self.network.LAMBDA*s))**2)
            weighted_errors = 0
            for j in range(len(errors)):
                w = self.network.layers[self.id+1].neurones[j].weights[i]
                e = errors[j]
                weighted_errors += w*e
            self.error.append(derivate*weighted_errors)
    def update_weights(self):
        for i in range(self.no):
            neurone = self.neurones[i]
            e = self.error[i]
            for j in range(neurone.n):
                if self.id > 0:
                    x = self.network.layers[self.id-1].neurones[j].last_output
                else:
                    x = self.network.input[j]
                neurone.weights[j] += self.network.ALPHA*e*x
                neurone.threshold += self.network.ALPHA*e
        
        
            
class Network():
    def __init__(self,dimensions):
        self.layers = []
        for i in range(1,len(dimensions)):
            self.layers.append(Layer(dimensions[i-1],dimensions[i],i-1,self))
        self.expected_output = None
        self.input = None
        self.last_output = None
        self.ALPHA = 1
        self.LAMBDA = 1
    def output(self,input):
        result = input
        for layer in self.layers:
            result = layer.output(result)
        self.last_output = result
        return(result)
    def backpropagate(self):
        for i in range(len(self.layers)-1,-1,-1):
            self.layers[i].retropropagate()
    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()
    def train(self,input,expected_output,alpha):
        self.ALPHA = alpha
        self.input = input
        self.expected_output = expected_output
        self.output(input)
        self.backpropagate()
        self.update_weights()
        

### Example

# On entraîne un réseau de neurones à trouver l'indice de l'élément maximum d'une liste

# test = Network([3,10,10,3]) 
# for i in range(10000):
#     input = [random() for j in range(3)]
#     expected_output = [0,0,0]
#     expected_output[input.index(max(input))] = 1
#     test.train(input,expected_output,0.1)
#     print(i)
# print(test.output([0.4,0.7,0.5]))

            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        