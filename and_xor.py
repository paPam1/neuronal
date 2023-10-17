import numpy as np


class Xarxa:
    def __init__(self, learning_rate=0.25):
        self.learning_rate = learning_rate

        self.pes_oc_ent = np.random.rand(3,2) * 0.1  # valores generados por funcion np están entre [0,1]; capa oculta  3 nodos y la capa entrada  2 nodos
        self.pesos_sort_oc = np.random.rand(2, 3) * 0.1  # capa  salida 2 nodos y capa oculta 3 nodos ;

        self.bia_oc = np.ones((3, 1))
        self.bia_sort = np.ones((2, 1))

    def sigmoid_d(self, x):  # calculamos gradiente
        return x * (1 - x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def error_calcul(self, salida, y):
        error = salida - y
        return error

    def backward_propagation(self, entradas, valor_deseado):
        # calculo de deltas para la capa de salida
        delta_a_salida = self.salida - valor_deseado
        delta_z_salida = delta_a_salida * self.sigmoid_d(self.salida)

        # actualizamos pesos y bias para la capa de salida
        delta_w_salida = np.dot(delta_z_salida, self.activacion_oculta.T)
        delta_b_salida = delta_z_salida
        self.pesos_sort_oc = self.pesos_sort_oc - self.learning_rate * delta_w_salida
        self.bia_sort = self.bia_sort - self.learning_rate * delta_b_salida

        # calculo de deltas para la capa oculta ; .T es traspuesta acuerdate!
        delta_z_oculta = np.dot(self.pesos_sort_oc.T, delta_z_salida) * self.sigmoid_d(self.activacion_oculta)  # np.dot producto
        delta_a_oculta = delta_z_oculta

        # actualizamos pesos y bias para la capa oculta
        delta_w_oculta = np.dot(delta_z_oculta, entradas.T)
        delta_b_oculta = delta_z_oculta
        self.pes_oc_ent = self.pes_oc_ent - self.learning_rate * delta_w_oculta
        self.bia_oc = self.bia_oc - self.learning_rate * delta_b_oculta

    def entreno_red(self, entradas, valor_deseado, veces=5000):
        for vez in range(veces):
            # para obtener salida red
            self.activacion_oculta = self.sigmoid(np.dot(self.pes_oc_ent, entradas) + self.bia_oc)
            self.salida = self.sigmoid(np.dot(self.pesos_sort_oc, self.activacion_oculta) + self.bia_sort)

            # calculo error
            error_c = self.error_calcul(self.salida, valor_deseado)  # lo que obtenemos - lo que queremos

            # propagacion
            self.backward_propagation(entradas, valor_deseado)


if __name__ == '__main__':
    xarxa = Xarxa()  # Aumentamos la tasa de aprendizaje

    # Salida deseada
    valor_deseado = np.array([[0], [1]])

    # Entradas [0, 1]
    entradas = np.array([0, 1]).reshape((2, 1))
    # Entreno la red
    #mirar como calcular and_zor xarxa.entreno_red(entradas, valor_deseado, veces=5000)  # Aumentamos el número de veces

    print(" ", xarxa.salida[0])  # salida 0
    print(" ", xarxa.salida[1])  # salida 1