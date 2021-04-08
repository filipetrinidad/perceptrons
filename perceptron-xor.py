import numpy as np

entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([[0], [1], [1], [0]])

pesos0 = np.array([[-0.345, -0.289, -0.134], [0.876, 0.387, -0.456]])
pesos1 = np.array([[0.765], [-0.234], [-0.492]])

treinos = 100
taxaAprendizado = 0.3
momentum = 1

def sigmoid(soma):
    return 1 / ( 1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1-sig)

derivada0 = sigmoidDerivada(0.5)
derivada1 = sigmoidDerivada(sigmoid)

for i in range(treinos):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - camadaSaida
    media = np.mean(np.abs(erroCamadaSaida))
    

