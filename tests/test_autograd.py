import numpy as np
import pytest
from mymllib.autograd import Tensor

def test_autograd_scalar_ops():
    x = Tensor(2.0, label='x')
    y = Tensor(-3.0, label='y')
    z = (x * y + 5.0) ** 2
    z.backward()
    
    # z = (2 * -3 + 5)^2 = (-1)^2 = 1.0
    assert np.isclose(z.data, 1.0)
    # dz/dx = 2 * (x * y + 5) * y = 2 * (-1) * (-3) = 6.0
    assert np.isclose(x.grad, 6.0)
    # dz/dy = 2 * (x * y + 5) * x = 2 * (-1) * 2 = -4.0
    assert np.isclose(y.grad, -4.0)

def test_autograd_matrix_ops():
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = Tensor(np.array([[2.0, 0.0], [1.0, 3.0]]))
    C = (A @ B).sum()
    C.backward()
    
    # dC/dA = grad @ B.T where grad is ones(2, 2)
    # dC/dB = A.T @ grad
    assert A.grad.shape == (2, 2)
    assert B.grad.shape == (2, 2)
    assert np.allclose(A.grad, np.ones((2, 2)) @ B.data.T)
    assert np.allclose(B.grad, A.data.T @ np.ones((2, 2)))

def test_autograd_activations():
    x = Tensor(0.0)
    y = x.tanh()
    y.backward()
    assert np.isclose(y.data, 0.0)
    assert np.isclose(x.grad, 1.0)

    x2 = Tensor(1.0)
    y2 = x2.relu()
    y2.backward()
    assert np.isclose(x2.grad, 1.0)

    x3 = Tensor(-1.0)
    y3 = x3.relu()
    y3.backward()
    assert np.isclose(x3.grad, 0.0)
