import random
import numpy as np
import math

#небольшая нейронка для решения задачи
class Network(object):
    def __init__(self, sizes,inner_function, inner_function_prime, output_function, output_derivative, l1=0, l2=0, init_dist=('uniform',-0.05,0.05)):

        self.l1 = l1
        self.l2 = l2
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_dist = init_dist
   
        if self.init_dist[0] == 'uniform':
            self.biases = [np.random.uniform(low=self.init_dist[1], high=self.init_dist[2], size=(y, 1)) for y in sizes[1:]]
            self.weights = [np.random.uniform(low=self.init_dist[1], high=self.init_dist[2], size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        elif self.init_dist[0] == 'normal':
            self.biases = [np.random.normal(loc=self.init_dist[1], scale=self.init_dist[2], size=(y, 1)) for y in sizes[1:]]
            self.weights = [np.random.normal(loc=self.init_dist[1], scale=self.init_dist[2], size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        elif self.init_dist[0] == 'beta':
            self.biases = [np.random.beta(self.init_dist[1], self.init_dist[2], size=(y, 1)) for y in sizes[1:]]
            self.weights = [np.random.beta(self.init_dist[1], self.init_dist[2], size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

        self.inner_function = inner_function
        self.inner_function_prime = inner_function_prime

        self.output_function = output_function
        self.output_derivative = output_derivative

    def feedforward(self, a):

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = self.inner_function(np.dot(w, a) + b)

        output = np.dot(self.weights[-1], a) + self.biases[-1]

        return self.output_function(output)

    def update_mini_batch(self, mini_batch, eta):

        #модуль изменения весов и смещений
        w_b_change = 0
        
        #заполняем будущие градиенты нулями
        #это показывает как все веса и смещения изменились только из-за mini_batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        eps = eta / len(mini_batch)

        w_b_change += eps*sum([np.sum(layer ** 2) for layer in nabla_w ])
        w_b_change += eps*sum([np.sum(layer ** 2) for layer in nabla_b ])

        if self.l2 != 0:
            w_b_change += self.l2 * sum([np.sum(layer ** 2) for layer in self.weights ])

        if self.l1 != 0:
            w_b_change += self.l1 * sum([np.sum(np.sign(layer) ** 2) for layer in self.weights ])


        self.weights = [w - eps * nw - self.l1 * np.sign(w) - self.l2 * w for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - eps * nb for b, nb in zip(self.biases,  nabla_b)]

        return math.sqrt(w_b_change)



    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # прямое распространение
        activation = x
        activations = [x]  # лист, хранящий все активации, слой за слоем
        zs = []  # лист, хранящий все z векторы, слой за слоем

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.inner_function(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        output = self.output_function(z)
        activations.append(output)

        # обратное распространение
        delta = (activations[-1] - y) * self.output_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        #обращаемся по негативному индексу - удобно
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.inner_function_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w