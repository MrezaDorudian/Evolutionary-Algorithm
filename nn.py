import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):

        self.w_matrix_layer_1 = np.random.normal(size=(layer_sizes[1], layer_sizes[0]))
        self.w_matrix_layer_2 = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))
        self.b_vector_layer_1 = np.zeros((layer_sizes[1], ))
        self.b_vector_layer_2 = np.zeros((layer_sizes[2], ))
        pass

    def activation(self, x):
        output = []
        for i in range(len(x)):
            output.append(1 / (1 + np.exp(-x[i])))
        return output

    def forward(self, x):
        z_vector_layer_1 = np.add(np.matmul(self.w_matrix_layer_1.copy(), x.copy()), self.b_vector_layer_1.copy())
        output_vector_layer_1 = self.activation(z_vector_layer_1.copy())
        # layer 1 output ready

        z_vector_layer_2 = np.add(np.matmul(self.w_matrix_layer_2.copy(), output_vector_layer_1.copy()),
                                  self.b_vector_layer_2.copy())
        output_vector_layer_2 = self.activation(z_vector_layer_2.copy())
        # layer 2 output ready

        return output_vector_layer_2
