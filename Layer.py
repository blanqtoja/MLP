# klasa ma przechowywac wiele neuronow - warstwa neuronow
#zarzadzamy propagacja w przod
#zarzadzamy propgacja wstecz (liczymy gradienty)
#przekazujemy bledy do poprzedniej warstwy (propagacja wsteczna)
#aktualizacja wszystkuch wag neuronow  wwarstwie
from Neuron import Neuron 
import random as r

class Layer:
    def __init__(self, numNeurons, numInputs , activationFun, activationFunDerivative, useBias = True):
        r.seed()

        self.delta = [0.0] * numNeurons
        self.neurons = []
        #tworzymy naurony w warstwie
        for _ in range(numNeurons):
            #losujemy wagi
            weigths =[r.uniform(-0.5, 0.5) for _ in range (numInputs)]
            bias = r.uniform(-0.5, 0.5) if useBias else 0.0 #jesli mamy uzywac bias, to losujemy
            self.neurons.append( Neuron( 
                [0.0] * numInputs, #na poczatku wejscia rownaja sie 0.0
                weigths,
                activationFun,
                activationFunDerivative,
                bias
            ))
            



    #liczy wyjscia kazdego neuronu, zwraca liste wyjsc
    def forwardPass(self, inputs):
        outputs = []
        for n in self.neurons:
            n.inputs = inputs #nadpisuje inputy 
            outputs.append(n.output()) # liczy outputy dla noweych inputow

        return outputs
        

    # def calcGradients():
    #     pass
    # def updateAllWeights(self, target, learningRate):
    #     for i, n in enumerate(self.neurons):
    #         n.updateWeights(target[i], learningRate)
    def updateAllWeights(self, learningRate):
        for n in self.neurons:
            n.updateWeights(learningRate)

#todo:
# sprawdz czy ok
    def backwardOutputLayer(self, targets):
        #liczy delty dla warsty wyjsciowej
        #potrzebne do rozpoczecia propagacji bledu
        #uzywaj neuron.calcDelta(target)
        delta = []
        for i, n in enumerate(self.neurons):
            delta.append( n.calcDelta(targets[i]) )
        self.delta = delta
        return delta
        

    def backwardHiddenLayer(self, nextLayer):
        #liczy delty dla warstwy ukrytej, bazuje na kolejnej warstwie
        # kontynuujemy propagacje wstecz
        #neuron.calcDeltaHIdden(nextLayer.neurons, myIndex)
        
        delta = []
        for i, n in enumerate(self.neurons):
            delta.append( n.calcDeltaHidden(nextLayer.neurons, i ))
        self.delta = delta
        return delta


