import value
import random
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [value.Value(random.uniform(-1, 1)) for _ in range(self.num_inputs)]
        self.bias = value.Value(random.uniform(-1, 1))

    
    def __call__(self, x):
        dot_product_sum = sum([wi*xi for wi,xi in zip(self.weights, x)]) + self.bias
        activated_val = dot_product_sum.tanh()
        return activated_val

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, prev_layer_size, num_neurons):
        self.prev_layer_size = prev_layer_size
        self.neurons = [Neuron(prev_layer_size) for _ in range(num_neurons)]
    
    def __call__(self, x):
        out_vals = [neuron(x) for neuron in self.neurons]
        return out_vals
    
    def __repr__(self):
        return f"Input Size: {self.prev_layer_size}\tLayer Size: {len(self.neurons)}"
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class Network:
    LR=0.01
    def __init__(self, nin, nouts):
        layer_sizes = [nin] + nouts
        self.layers = [Layer(layer_sizes[index], layer_sizes[index+1]) for index in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def grad_step(self):
        for param in self.parameters():
            param.data -= param.grad*Network.LR
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

def mse_loss(labels, preds):
    loss = sum((li - pi)**2 for li, pi in zip(labels, preds))
    assert(isinstance(loss, value.Value))
    return loss



if __name__ == '__main__':
    ff_nn = Network(3, [4, 5, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    running_loss = []
    for _ in range(100):
        preds = [ff_nn(x) for x in xs]
        loss = mse_loss(ys, preds)
        running_loss.append(loss.data)
        loss.backward()
        #print(f"GRAD: {ff_nn.layers[0].neurons[0].weights[0].grad}")
        #print(f"WEIGHT: {ff_nn.layers[0].neurons[0].weights[0].data}")
        ff_nn.grad_step()
        #print(f"WEIGHT AFTER: {ff_nn.layers[0].neurons[0].weights[0].data}")
        ff_nn.zero_grad()
        #print(f"GRAD AFTER: {ff_nn.layers[0].neurons[0].weights[0].grad}")
    
    print(running_loss[-1])
    print([ff_nn(x) for x in xs])
