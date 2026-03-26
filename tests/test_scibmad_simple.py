import pytest
from scibmad import core as scibmad 
import numpy as np
import torch


@pytest.fixture(scope="session", autouse=True)
def setup_julia():
    """Define Julia test functions once for the whole session."""
    scibmad.seval("""
    f_square(x) = x^2
    f_sum(arr) = sum(arr)
    f_linear(arr) = 2.0 .* arr
    f_multi_output(x) = [x, x^2, x^3]
    """)


# calling julia from python

def test_scalar_function():
    """A simple Julia scalar function should return the correct value"""
    x = torch.tensor(3.0, dtype=torch.float64)
    result = float(scibmad.f_square(x))
    assert result == pytest.approx(9.0)


def test_array_sum():
    """Julia sum over a torch tensor should match numpy"""
    arr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = float(scibmad.f_sum(arr))
    assert result == pytest.approx(10.0)


def test_array_output():
    """Julia function returning an array should be iterable from Python"""
    arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = np.array([float(v) for v in scibmad.f_linear(arr)])
    expected = np.array([2.0, 4.0, 6.0])
    np.testing.assert_allclose(result, expected)


def test_multi_output():
    """Julia function returning multiple values should unpack correctly"""
    x = torch.tensor(2.0, dtype=torch.float64)
    result = [float(v) for v in scibmad.f_multi_output(x)]
    assert len(result) == 3
    assert result == pytest.approx([2.0, 4.0, 8.0])


def test_result_is_finite():
    """Output should never be NaN or Inf for normal inputs"""
    arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = np.array([float(v) for v in scibmad.f_linear(arr)])
    assert np.all(np.isfinite(result))


def test_deterministic():
    """Same input should always produce the same output"""
    x = torch.tensor(4.0, dtype=torch.float64)
    r1 = float(scibmad.f_square(x))
    r2 = float(scibmad.f_square(x))
    assert r1 == r2


# autograd through julia

def test_gradient_scalar():
    """Gradient of x^2 should be 2x"""
    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    result = scibmad.f_square(x)
    result.backward()
    assert x.grad is not None
    assert x.grad == pytest.approx(6.0)


def test_gradient_is_finite():
    """Gradients should be finite, not NaN or Inf"""
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    result = scibmad.f_square(x)
    result.backward()
    assert torch.isfinite(x.grad)


def test_gradient_array():
    """Gradients should flow back through array inputs"""
    arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = scibmad.f_sum(arr)
    result.backward()
    assert arr.grad is not None
    np.testing.assert_allclose(arr.grad.numpy(), np.ones(3))