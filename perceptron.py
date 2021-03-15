import numpy as np
#perceptron, a layer 

entrada = np.array([1, 5, 8])
pesos = np.array([1, 0.8, 0])

def funcaosoma(e, p):
    return e.dot(p)

soma = funcaosoma(entrada, pesos)

def stepfunction(s):
    if(s > 1):
        return 1
    else:
        return 0

saida = stepfunction(soma)
print(saida)