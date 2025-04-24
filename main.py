from Network import Network
from Neuron import Neuron 
import math 
import random as r
from ucimlrepo import fetch_ucirepo 

def activationFunction(x):
    return 1 / ( 1 + math .exp(-x))

def activFunDer(x):
    return 
  


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)


# Dane XOR
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Tworzymy sieć
mlp = Network([2, 6, 1], activationFun=sigmoid, activationFunDerivative=sigmoid_derivative)

# Trenujemy
mlp.train(training_data, learningRate=0.2, epochs=20000)

# Testujemy
for input_vec, expected in training_data:
    pred = mlp.predict(input_vec)
    print(f"Wejście: {input_vec}, Oczekiwane: {expected}, Predykcja: {pred}")

test_inputs = [
    ([0.2, 0.8], "oczekiwane: około 1"),
    ([0.9, 0.9], "oczekiwane: około 0"),
    ([0.1, 0.1], "oczekiwane: około 0"),
    ([0.5, 0.5], "oczekiwane: około 0.2 - 0.3"),
]

for inp, desc in test_inputs:
    pred = mlp.predict(inp)
    print(f"Wejście: {inp}, {desc}, Predykcja: {pred}")
