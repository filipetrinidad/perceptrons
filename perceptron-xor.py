import numpy as np

entradas = np.array([0,0], [0,1], [1,0], [1,1])
saidas = np.array([0], [1], [1], [0])

pesos0 = np.array([-0.345, -0.289, -0.134, 0.876, 0.387, -0.456])
pesos1 = np.array([[0.765], [-0.234], [-0.492])

treinos = 1000
taxaaprendizado = 0.3
momentum = 1

def sigmoide(soma):
    return 1 / ( 1 + np.exp(-soma))

