import numpy as np
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = set(_children)
        self._operation= _op
        self.grad = 0.0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, b):
        b = b if isinstance(b, Value) else Value(b)
        out = Value(self.data + b.data, (self, b), '+')

        def backward():
            self.grad += out.grad * 1
            b.grad += out.grad * 1

        out._backward = backward
        return out
    
    def __mul__(self, b):
        b = b if isinstance(b, Value) else Value(b)
        out = Value(self.data * b.data, (self, b), '*')

        def backward():
            self.grad += out.grad * b.data
            b.grad += out.grad * self.data

        out._backward = backward
        return out
    
    def __rmul__(self, b):
        return self * b
    
    def __radd__(self, b):
        return self + b
    
    def __rsub__(self, b):
        return self - b
    
    def __sub__(self, b):
        b = b if isinstance(b, Value) else Value(b)
        out = Value(self.data - b.data, (self, b), '-')

        def backward():
            self.grad += out.grad * 1
            b.grad += out.grad * 1
            
        out._backward = backward
        return out
    
    def __pow__(self, b):
        assert isinstance(b, (int, float)), "Int and Float supported only for pow(Value, n)"
        out = Value(self.data**b, (self,), '**')

        def backward():
            self.grad += out.grad * (b * self.data**(b-1))

        out._backward = backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        
        def backward():
            self.grad += (1 - t**2) * out.grad

        out = Value(t, (self,), 'tanh')
        out._backward = backward
        return out
    
    def backward(self):

        postordered = []
        visited = set()
        def postorder(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    postorder(child)
                postordered.append(node)
        
        self.grad = 1.0
        postorder(self)
        for node in reversed(postordered):
            node._backward()
                    



if __name__ == '__main__':
    a = Value(-4)
    b = Value(-3)
    c = a * b
    d = c + 5
    e = c**2
    e.backward()
    print(a.grad)
    
