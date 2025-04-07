from neuron import Neuron 
import math 
import random as r
from ucimlrepo import fetch_ucirepo 

def activationFunction(x):
    return 1 / ( 1 + math .exp(-x))

  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
# print(iris.metadata) 
  
# variable information 
# print(iris.variables) 
print(X)


r.seed()
# print(r.random() - 0.5)
bias = 0
rand = r.random() - 0.5

# n1 = Neuron([1, 2], [r.random() - 0.5, r.random() - 0.5], activationFunction, bias)
# print(n1.output())

inputsWeights = []
for i in range( len(X) ):
    inputsWeights.append(r.random() - 0.5)

# print(inputsWeights)
T = []
T.append(Neuron(X, inputsWeights, activationFunction, 0))

outputs = []
for t in T:
    outputs.append(t.output())
    
print(outputs)