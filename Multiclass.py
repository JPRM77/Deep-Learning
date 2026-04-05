import numpy as np

def sigmoid (x):
        return 1/(1 + np.exp(-x))

def softmax (x):
        e_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return e_x/e_x.sum(axis = 1, keepdims = True)

class multiclass_detector:
        def __init__ (self, input_size, hidden_size, output_size):
                self.w1 = np.random.uniform(-1, 1, size = (input_size, hidden_size))
                self.b1 = np.random.uniform(-1, 1, size = (1, hidden_size))
                self.z1 = np.zeros((1, hidden_size))
                self.a1 = np.zeros((1, hidden_size))


                self.w2 = np.random.uniform(-1, 1, size = (hidden_size, output_size))
                self.b2 = np.random.uniform(-1, 1, size = (1, output_size))
                self.z2 = np.zeros((1, output_size))
                self.a2 = np.zeros((1, output_size))

        def foward_propagation (self, x):
                self.z1 = x@self.w1 + self.b1
                self.a1 = sigmoid(self.z1)

                self.z2 = self.a1@self.w2 + self.b2
                self.a2 = softmax(self.z2)

                return self.a2



        def backpropagation(self, inputs, targets, learning_rate):
                inputs = np.atleast_2d(inputs)
                targets = np.atleast_2d(targets)

                delta_2 = self.a2 - targets

                delta_1 = (delta_2@self.w2.T)*(self.a1*(1 - self.a1))


                self.w2 -= (self.a1.T@delta_2)*learning_rate
                self.b2 -= np.sum(delta_2, axis = 0, keepdims = True)*learning_rate

                self.w1 -= (inputs.T@delta_1)*learning_rate
                self.b1 -= np.sum(delta_1, axis = 0, keepdims = True)*learning_rate

        def train (self, inputs, targets, learning_rate, epochs):
                for i in range(epochs):
                        for k, j in zip(inputs, targets):
                                self.foward_propagation(k)
                                self.backpropagation(k, j, learning_rate)


if __name__ == '__main__':
        neuron = multiclass_detector(3, 3, 3)

        inputs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        targets = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        neuron.train(inputs, targets, 0.1, 10000)

        x1 = float(input('Digita qual é a primeira entrada: '))
        x2 = float(input('Digite qual é a segunda entrada: '))
        x3 = float(input('Digite qual é a terceira entrada: '))
        x = np.array([[x1, x2, x3]])

        y = neuron.foward_propagation(x)

        print('O resultado obtiodo foi: {}'.format(np.round(y, decimals = 2)))
