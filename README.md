# scibmad

`scibmad` is a bridge between PyTorch and Julia for differentiable scientific workflows. It lets you define or load Julia functions, call them from Python, and keep gradients flowing through PyTorch autograd.

## What it provides

- PyTorch tensor to Julia value conversion
- Autograd-aware wrappers for Julia functions
- Beamline helpers for SciBmad element tracking
- A Julia-style object API for screenshot-style beamline workflows
- A small Python API for loading Julia code into `Main`

## Requirements

- Python 3.11+
- Julia 1.9+
- A working SciBmad-compatible Julia environment

## Installation

Install the package:

```bash
pip install git+https://github.com/ChanpiseyU/scibmad.git
```

Core Python dependencies:

```bash
pip install torch juliacall numpy
```

If your Julia environment does not already include the required packages, `scibmad` will bootstrap `SciBmad` during initialization.

## Quick Start

```python
from scibmad import core as scibmad
import torch

scibmad.define(
    """
    f_square(x) = x^2
    f_scaled_sum(scale, arr) = scale * sum(arr)
    """
)

x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
y = scibmad.f_square(x)
y.backward()

print(float(y))      # 9.0
print(float(x.grad)) # 6.0
```

## Beamline Example

The recommended beamline interface mirrors Julia SciBmad closely:

```python
import scibmad as sb
import torch

# Make a FODO cell
qf = sb.Quadrupole(Kn1=0.36, L=0.5)
d = sb.Drift(L=1.2)
qd = sb.Quadrupole(Kn1=-0.36, L=0.5)

# 18 GeV electrons
fodo = sb.Beamline(
    [qf, d, qd, d],
    E_ref=18e9,
    species_ref=sb.Species("electron"),
)

coords = torch.tensor(
    [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
    dtype=torch.float64,
    requires_grad=True,
)

qf.Kn1 = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
qd.Kn1 = -qf.Kn1

result = sb.track(fodo, v0=coords)
loss = result.v[0, :4, -1].sum()
loss.backward()
```

`scibmad` also keeps the lower-level functional helpers for direct element-by-element tracking:

```python
from scibmad import core as scibmad
import torch

coords = torch.tensor(
    [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0],
    dtype=torch.float64,
    requires_grad=True,
)

line = scibmad.beamline(
    scibmad.drift(1.0, p_over_q_ref=2.0),
    scibmad.quadrupole(0.5, 1.2, p_over_q_ref=2.0),
)

result = line(coords)
loss = result.sum()
loss.backward()
```

## Public API

- `scibmad.define(code)` evaluates Julia code in `Main`
- `scibmad.load(path)` includes a Julia source file
- `scibmad.seval(code)` evaluates arbitrary Julia expressions
- `scibmad.using(*packages)` loads Julia packages with `using`
- `scibmad.Quadrupole(...)`, `scibmad.Drift(...)`, `scibmad.Beamline(...)`, and `scibmad.track(...)` provide a Julia-style beamline API
- Julia functions defined in `Main` become available as `scibmad.<name>`

## Development

Run the focused test suite with:

```bash
python run_tests.py
```

## References

- [Julia](https://julialang.org/)
- [JuliaCall](https://pypi.org/project/juliacall/)
- [PyTorch](https://pytorch.org/)
- [SciBmad](https://github.com/bmad-sim/SciBmad.jl)

## License

MIT
