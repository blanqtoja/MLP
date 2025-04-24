# Chcemy dobrać wagi tak, aby funkcja kosztu była jak najmniejsza,
# czyli przeciwnie do gradientu:

#siec zarzadza warstwami
#jest odpowiedzialna 
# za forward i backward
# aktualizacja wag wszystkich neurownow
# umozliwia trenowanie pojedynczego wzorca

from Layer import Layer
import json

class Network:
    #layerStructure - okresla ilosc neurownow w poszczegolnych warstwarch np [3,7,2] - input 3, hidden 7, output 2
    def __init__(self, layerStructure, activationFun, activationFunDerivative, useBias = True):
        self.layers = []
        #wypelnienie sieci warstwami z odpowiednia iloscia neuronow
        for i in range(1, len(layerStructure)):
            self.layers.append( Layer(
                numNeurons = layerStructure[i],
                numInputs = layerStructure[i-1],
                activationFun = activationFun,
                activationFunDerivative = activationFunDerivative,
                useBias = useBias
            ))

    def forward(self, inputs):
        for layer in self.layers:
            #nadpisanie wejscia przez metode forwardPass z parametrem input
            #dla wszystkich warstw
            inputs = layer.forwardPass(inputs)
        return inputs
    
    #targets - wartosci oczekiwane
    def backward(self, targets, learningRate):
        #[-1] oznacza ostatni element listy
        #on musi wykorzystac backward dla outputlayer, bo jest warstwa wyjsciowa
        self.layers[-1].backwardOutputLayer(targets)

        #idziemy od tylu po liscie warstw
        #wykorzystujemy backward dla hidden
        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].backwardHiddenLayer(self.layers[i+1])

        #teraz nastepuje aktualizacja wag 
        for i in range(len(self.layers)):
            #jesli i jest rowny ostatniemu elementowi, jesli nie to dla kazdego neuronu z warstw (bez wejsciowej) to wyciagamy outputy nastepnej wasrtwy (wyzszej)
            # self.layers[i].updateAllWeights(targets if i == len(self.layers)-1 else [n.output() for n in self.layers[i+1].neurons], learningRate)
            self.layers[i].updateAllWeights(learningRate)

    def train(self, trainingData, learningRate=0.1, epochs=1000):
        # for _ in range(epochs):
        #     for inputs, targets in trainingData:
        #         self.forward(inputs)
        #         self.backward(targets, learningRate)
        for epoch in range(epochs):
            loss = 0
            for inputs, targets in trainingData:
                outputs = self.forward(inputs)
                loss += sum((t - o)**2 for t, o in zip(targets, outputs))
                self.backward(targets, learningRate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                

    def predict(self, inputs):
        return self.forward(inputs)

    def saveToFile(self, filename):
        network_data = []

        for layer in self.layers:
            layer_data = []
            for neuron in layer.neurons:
                neuron_data = {
                    'weights': neuron.inputWeights,
                    'bias': neuron.bias
                }
                layer_data.append(neuron_data)
            network_data.append(layer_data)

        with open(filename, 'w') as file:
            json.dump(network_data, file, indent=4)

    def loadFromFile(self, filename, activationFun, activationFunDerivative, useBias=True):
        with open(filename, 'r') as file:
            network_data = json.load(file)

        self.layers = []
        for i, layer_data in enumerate(network_data):
            numInputs = len(layer_data[0]['weights'])  #liczba wejsc na neuron
            layer = Layer(
                numNeurons=len(layer_data),
                numInputs=numInputs,
                activationFun=activationFun,
                activationFunDerivative=activationFunDerivative,
                useBias=useBias
            )
            for j, neuron_data in enumerate(layer_data):
                layer.neurons[j].inputWeights = neuron_data['weights']
                layer.neurons[j].bias = neuron_data['bias']
            self.layers.append(layer)
