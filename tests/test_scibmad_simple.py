import pytest
import scibmad as scibmad_pkg
from scibmad import core as scibmad 
import numpy as np
import torch


@pytest.fixture(scope="session", autouse=True)
def setup_julia():
    """Define Julia test functions once for the whole session."""
    scibmad.define("""
    f_square(x) = x^2
    f_sum(arr) = sum(arr)
    f_linear(arr) = 2.0 .* arr
    f_multi_output(x) = [x, x^2, x^3]
    f_scaled_sum(scale, arr) = scale * sum(arr)
    f_dot_sum(a, b) = sum(a .* b)
    """)


# calling julia from python

def test_scalar_function():
    """A simple Julia scalar function should return the correct value"""
    x = torch.tensor(3.0, dtype=torch.float64)
    result = float(scibmad.f_square(x))
    assert result == pytest.approx(9.0)


def test_top_level_julia_function():
    """Julia functions should be available as scibmad.<function_name>."""
    x = torch.tensor(3.0, dtype=torch.float64)
    result = float(scibmad_pkg.f_square(x))
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


def test_gradient_with_fixed_arg():
    """Fixed Julia/Python args should pass through while tensor args get gradients."""
    arr = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = scibmad.f_scaled_sum(2.5, arr)
    result.backward()
    assert arr.grad is not None
    np.testing.assert_allclose(arr.grad.numpy(), np.full(3, 2.5))


def test_gradient_with_multiple_tensor_args():
    """Multiple tensor arguments should each receive gradients."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64, requires_grad=True)
    result = scibmad.f_dot_sum(a, b)
    result.backward()
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.numpy(), b.detach().numpy())
    np.testing.assert_allclose(b.grad.numpy(), a.detach().numpy())


def test_reference_rigidity_positional_arg_keeps_autograd_path():
    """p_over_q_ref can be passed positionally while tensor coordinates stay differentiable."""
    coords = torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = scibmad.quadrupole(coords, 0.5, 1.2, 2.0)
    loss = result.sum()
    loss.backward()

    assert coords.grad is not None
    assert torch.all(torch.isfinite(coords.grad))


def test_beamline_elements_accept_reference_rigidity():
    """Element constructors can store p_over_q_ref for differentiable beamline tracking."""
    coords = torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    beamline = scibmad.beamline(
        scibmad.drift(1.0, p_over_q_ref=2.0),
        scibmad.quadrupole(0.5, 1.2, p_over_q_ref=2.0),
    )

    result = beamline(coords)
    loss = result.sum()
    loss.backward()

    assert coords.grad is not None
    assert torch.all(torch.isfinite(coords.grad))


def test_object_style_api_tracks_and_keeps_autograd():
    """A thin object-style API should support screenshot-style optimization."""
    qf = scibmad.Quadrupole(Kn1=0.36, L=0.5)
    d = scibmad.Drift(L=1.2)
    qd = scibmad.Quadrupole(Kn1=-0.36, L=0.5)
    line = scibmad.Beamline(
        [qf, d, qd, d],
        E_ref=18e9,
        species_ref=scibmad.Species("electron"),
    )

    coords = torch.tensor(
        [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    k1 = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    qf.Kn1 = k1
    qd.Kn1 = -k1
    result = scibmad.track(line, v0=coords)
    loss = result.v[0, :4, -1].sum()
    loss.backward()

    assert result.v.shape == (1, 6, 2)
    assert k1.grad is not None
    assert torch.isfinite(k1.grad)
