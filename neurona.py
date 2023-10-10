import sys
import numpy as np

class neuro:
    def __init__(self):


        self.bia_so = np.ones((2, 1))  # Bias salida
        self.bia_hi = np.ones((3, 1))  # Bias oculta
        self.weight_hi_en = np.ones((3, 2))  # Pesos entrada oculta
        self.weight_hi_so = np.ones((2, 3))  # Pesos oculta asalida


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, entrades):
        # Cálculo oculta
        suma_weight = np.dot(self.weight_hi_en, entrades) + self.bia_hi
        activation = self.sigmoid(suma_weight)

        # Cálculo salida
        sortida_suma_weight = np.dot(self.weight_hi_so, activation) + self.bia_so
        salida = self.sigmoid(sortida_suma_weight)

        return salida

if __name__ == '__main__':
    functn = neuro()

    info = np.array([0, 1]).reshape((2, 1))

    salida = functn.feedforward(info)

    print("Sortida :", salida)
