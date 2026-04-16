import random
from autograd import Value
from typing import List

class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.random()) for _ in range(nin)]
        self.b = Value(0.0)
        
    def __call__(self, x):
        activation = sum((w1*x1 for w1, x1 in zip(self.w, x)), self.b)
        return activation.tanh()
    
class Layer:
    
    def __init__(self, nin, nout):
        self.layer = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        return [n(x) for n in self.layer]
        
class MLP:
    
    def __init__(self, layers: List[int]): # Input, Hidden, Output       
        self.layers = [Layer(nin,nout) for nin, nout in zip(layers, layers[1:])]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x