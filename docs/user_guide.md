
## Legendre Polynomial

``` python
from scibmad import core as scibmad
import torch
import math

scibmad.define("""
function legendre_p3(x)
    if x isa AbstractArray
        return 0.5 .* (5 .* x.^3 .- 3 .* x)
    else
        return 0.5 * (5 * x^3 - 3 * x)
    end
end
""")

x = torch.linspace(-math.pi, math.pi, 2000, dtype=torch.float64)
y = torch.sin(x)

a = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
b = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
c = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
d = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.SGD([a, b, c, d], lr=5e-6)

for i in range(2000):
    optimizer.zero_grad()
    y_pred = a + b * scibmad.legendre_p3(c + d * x)
    loss = (y_pred - y).pow(2).sum()
    loss.backward()
    optimizer.step()

```

# Legendre Polynomial Results 
```
Final loss: 9.090238e+00
Optimized: a=-0.0000, b=-2.1965, c=0.0000, d=0.2554
```

## Bessel-Like Function Approximation

``` python
from scibmad import core as scibmad
import torch
import math

scibmad.define("""
function bessel_approx(x)
    if x isa AbstractArray
        x2 = x.^2
        return 1 .- x2 ./ 4 .+ x2.^2 ./ 64 .- x2.^3 ./ 2304
    else
        x2 = x^2
        return 1 - x2 / 4 + x2^2 / 64 - x2^3 / 2304
    end
end
""")

x3 = torch.linspace(0, 5, 500, dtype=torch.float64)
y3 = 1 - (x3**2) / 4 + (x3**4) / 64 - (x3**6) / 2304
y3 += 0.01 * torch.randn_like(y3)

scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.Adam([scale, shift], lr=1e-2)

```

# Bessel-Like Function Approximation Results 
```
Final loss: 0.0479
Optimized: scale=0.9993, shift=0.0032
```

## FODO Quadrupole Error Optimization 

``` python
from scibmad import core as scibmad
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

rng = np.random.default_rng(42)
true_errors_np = rng.uniform(-0.5, 0.5, size=2)

model_kn1_np = np.array([5.0, -5.0], dtype=np.float64)
real_kn1_np = model_kn1_np + true_errors_np

coords0 = torch.tensor([1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0], dtype=torch.float64)
target = torch.tensor(
    real_kn1_np,
    dtype=torch.float64,
)

def make_line(kn1):
    return scibmad.beamline(
        scibmad.quadrupole_ele(0.5, kn1[0]),
        scibmad.drift_ele(1.0),
        scibmad.quadrupole_ele(0.5, kn1[1]),
        scibmad.drift_ele(1.0),
    )

def track_particle(coords, kn1):
    line = make_line(kn1)
    out = line.track(coords)
    return out[:4]

with torch.no_grad():
    target = track_particle(coords0, torch.tensor(real_kn1_np, dtype=torch.float64))

q1 = torch.tensor([model_kn1_np[0]], requires_grad=True)
q2 = torch.tensor([model_kn1_np[1]], requires_grad=True)

optimizer = torch.optim.Adam([q1, q2], lr=1e-3)
iterations = 10000

for i in range(iterations):
    optimizer.zero_grad()

    kn1 = torch.cat([q1, q2])
    result = track_particle(coords0, kn1)
    loss = ((result - target) ** 2).sum()

    loss.backward()
    optimizer.step()


```

# FODO Quadrupole Error Optimization Results 

| Quad | Model  | Real (erroneous) | Optimized   |
|------|--------|------------------|-------------|
| QF   | 5.0000 | 5.27395605       | 5.27395605  |
| QD   | -5.0000| -5.06112156      | -5.06112156 |

| Quad | True Error  | Optimized Error | Difference |
|------|-------------|-----------------|------------|
| QF   | 0.27395605  | 0.27395605      | 0.00000000 |
| QD   | -0.06112156 | -0.06112156     | 0.00000000 |