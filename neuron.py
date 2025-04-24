# Implementacja perceptronu ma mieć charakter uniwersalny,
#  a zatem:
#  umożliwiający łatwą skalowalność jego architektury.
#  Oznacza to, że kod powinien :
# gwarantować poprawność działania i nauki perceptronu bez względu na liczbę warstw i neuronów w poszczególnych warstwach, które to 
# parametry powinny być określane w momencie tworzenia sieci.
#  Neurony przetwarzające, czyli neurony ukryte i wyjściowe (ewentualne neurony wejściowe nie są zaliczane do neuronów przetwarzających) 
#       mają mieć charakter nieliniowy 
#       i wykorzystywać sigmoidalną funkcję aktywacji, 
#           której współczynnik nachylenia ma być równy 1. 
#   Program ma umożliwiać określenie, czy podczas obliczania pobudzenia neuronu (sumy ważonej jego wejść) neurony przetwarzające mają uwzględniać wartość wejścia obciążającego (ang. bias), czy też nie.
#  Wagi sieci, o ile nie jest ona wczytywana z pliku, mają być inicjalizowane w sposób pseudolosowy wartościami z niewielkiego przedziału otaczającego 0 (np. z przedziału [-0,5; 0,5] lub [-1; 1]). 
# 
# Program ma umożliwiać zachowanie sieci do pliku oraz wczytanie z pliku zapisanej w nim sieci. Ma także umożliwiać wczytanie zestawu wzorców z pliku.

class Neuron:

    def __init__(self, inputs, inputWeights, activationFunction, activFunDer, bias):
        self.inputs = inputs
        self.inputWeights = inputWeights
        self.bias = bias
        self.activationFunction = activationFunction
        self.activFunDer = activFunDer
        # self.learningRate = 0.0
        self.delta = 0 #bufor na przechowywanie bleduz porpagacji wstecznej

    def  wSumInputs(self):
        sum = 0.0
        #dla kazdego wejscia: wejscie * waga + bias
        for i in range(len(self.inputs)): 
            sum += self.inputs[i] * self.inputWeights[i] 
        return sum + self.bias
    
    #forward pass
    def output(self):
        return self.activationFunction( self.wSumInputs())
    

    # d = z - y
    # blad predykcji
    def calcDiff(self, target):
        return target - self.output()

    def calcDelta(self, target):
        self.delta = (target - self.output()) * self.activFunDer(self.output())
        return self.delta

    
    def calcDeltaHidden(self, nextLayerNeurons, myIndex):
        sum = 0.0
        for neuron_k in nextLayerNeurons:
            delta_k = neuron_k.delta
            w_jk = neuron_k.inputWeights[myIndex]
            sum += delta_k * w_jk

        self.delta = self.activFunDer(self.output()) * sum
        return self.delta


    def gradientForEachW (self, target):
        # y = self.output()
        # d = self.calcDiff(target)
        # dy = self.activFunDer(y)

        delta = -self.calcDelta(target)
        return [ delta * x for x in self.inputs ]



    #do obliczenia nowych wag potrzebne są 
    # wejscia - self
    # target
    # learning rate - self 
    # def updateWeights(self, target, learningRate):
    #     gradients = self.gradientForEachW(target)
    #     for i in range( len(self.inputWeights)):
    #         self.inputWeights[i] += learningRate * gradients[i]
    #     self.bias += learningRate * (-self.delta)
    def updateWeights(self, learningRate):
        for i in range(len(self.inputWeights)):
            self.inputWeights[i] += learningRate * self.delta * self.inputs[i]
        self.bias += learningRate * self.delta

