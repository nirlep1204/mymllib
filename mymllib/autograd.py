import numpy as np
from typing import Tuple, Set, Callable, Union, List

def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad
    ndims_added = len(grad.shape) - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data: Union[int, float, list, np.ndarray], _children: tuple = (), _op: str = '', label: str = '') -> None:
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape})"

    def backward(self) -> None:
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()
        def build_topo(v: 'Tensor') -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self + other

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(-out.grad, other.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self * other

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward() -> None:
            self.grad += unbroadcast(out.grad / other.data, self.shape)
            other.grad += unbroadcast(-self.data * out.grad / (other.data ** 2), other.shape)
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, other: Union[int, float]) -> 'Tensor':
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward() -> None:
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Tensor':
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        val = np.clip(self.data, -500, 500)
        s = 1.0 / (1.0 + np.exp(-val))
        out = Tensor(s, (self,), 'sigmoid')
        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward() -> None:
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
        
    def exp(self) -> 'Tensor':
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward() -> None:
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
        
    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data + 1e-15), (self,), 'log')
        def _backward() -> None:
            self.grad += out.grad / (self.data + 1e-15)
        out._backward = _backward
        return out
        
    def sum(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')
        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            n = np.prod(self.data.shape) if axis is None else np.prod([self.data.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
            self.grad += grad * np.ones_like(self.data) / n
        out._backward = _backward
        return out

    def max(self, axis: Union[int, None] = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')
        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
            # divide grad if multiple max elements exist
            mask = mask / np.sum(mask, axis=axis, keepdims=True)
            self.grad += mask * grad
        out._backward = _backward
        return out

def trace(root: Tensor) -> Tuple[Set[Tensor], Set[Tuple[Tensor, Tensor]]]:
    """Builds a set of all nodes and edges in a graph."""
    nodes, edges = set(), set()
    def build(v: Tensor) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root: Tensor):
    """
    Draws the computational graph using Graphviz.
    For Tensors, it displays the matrix shapes rather than raw values to avoid clutter.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("Please install graphviz (pip install graphviz) to visualize the graph.")
        
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        # Format the label to show data and grad
        # If it's a scalar (shape is empty tuple), show the raw float value like Micrograd!
        # If it's a matrix, show the shape so the graph doesn't get flooded with giant arrays.
        data_str = f"{n.data.item():.4f}" if n.shape == () else f"shape {n.shape}"
        grad_str = f"{n.grad.item():.4f}" if n.shape == () else f"shape {n.grad.shape}"
        
        lbl = f"{n.label} | " if hasattr(n, 'label') and n.label else ""
        label = f"{{{lbl}data {data_str} | grad {grad_str}}}"
        
        dot.node(name=uid, label=label, shape='record')
        
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot
