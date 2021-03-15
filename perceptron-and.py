import numpy as np 

entradas = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
saidas = np.array([0, 0, 0, 1])
pesos = np.array([0.0, 0.0])

taxaAprendizado = 0.5

def soma(e, p):
    return e.dot(p)

s = soma(entradas, pesos)

def stepFunction(s):
    if (s >= 1):
        return 1
    else:
        return 0

def calculoSaida(reg):
    s = reg.dot(pesos)
    return stepFunction(s)

def aprendeAtualiza():
    erroTotal = 1
    while( erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            calcSaida = calculoSaida(np.array(entradas[i]))
            error = abs(saidas[i] - calcSaida)
            erroTotal += error
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizado * entradas[i][j] * error)
                print('pesos atualizados' + str(pesos[j]))
            print('total de erros' + str(erroTotal))

aprendeAtualiza()