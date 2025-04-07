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

    def __init__(self, inputs, inputWeights, activationFunction, bias):
        self.inputs = inputs
        self.inputWeights = inputWeights
        self.bias = bias
        self.activationFunction = activationFunction

    def  wSumInputs(self):
        sum = 0.0
        #dla kazdego wejscia: wejscie * waga + bias
        for i in range(len(self.inputs)): 
            sum += self.inputs[i] * self.inputWeights[i] 
        return sum + self.bias
    
    def output(self):
        return self.activationFunction( self.wSumInputs())