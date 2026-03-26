
## Legendre Polynomial

``` python
from scibmad import core as scibmad
import torch
import math

scibmad.seval("""
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

a = torch.tensor(0.0,  dtype=torch.float64, requires_grad=True)
b = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
c = torch.tensor(0.0,  dtype=torch.float64, requires_grad=True)
d = torch.tensor(0.3,  dtype=torch.float64, requires_grad=True)

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
Final loss: 8.9436
Optimized: a=-0.0000, b=-2.2085, c=0.0000, d=0.2555
```

## Bessel-Like Function Approximation

``` python 
from scibmad import core as scibmad
import torch
import math

scibmad.seval("""
function bessel_approx(x)
    if x isa AbstractArray
        x2 = x.^2
        return 1 .- x2./4 .+ x2.^2 ./64 .- x2.^3 ./2304
    else
        x2 = x^2
        return 1 - x2/4 + x2^2/64 - x2^3/2304
    end
end
""")

x3 = torch.linspace(0, 5, 500, dtype=torch.float64)
y3 = 1 - (x3**2)/4 + (x3**4)/64 - (x3**6)/2304
y3 += 0.01 * torch.randn_like(y3)

scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

optimizer1 = torch.optim.Adam([scale, shift], lr=1e-2)

for i in range(300):
    optimizer1.zero_grad()
    y_pred = scibmad.bessel_approx(scale * x3 + shift)
    loss1 = (y_pred - y3).pow(2).sum()
    loss1.backward()
    optimizer1.step()

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


scibmad.seval("""
using SciBmad
using ForwardDiff
using Random

const backend = AutoForwardDiff()
using DifferentiationInterface: gradient, jacobian, derivative

Random.seed!(42) #reproducibility, check other seeds to see different error patterns
errors = -0.5 .+ 1.0 .* rand(2)  # 2 errors: one for QF, one for QD

#MODEL
qf_mod = Quadrupole(L = 0.5, Kn1 =  5.0)
qd_mod = Quadrupole(L = 0.5, Kn1 = -5.0)

fodo_mod = [qf_mod, Drift(L=1.0), qd_mod, Drift(L=1.0)]
beam_mod = Beamline(fodo_mod, R_ref=-59.52872449027632, species_ref=Species("electron"))

#REAL
qf_real = Quadrupole(L = 0.5, Kn1 =  5.0 + errors[1])
qd_real = Quadrupole(L = 0.5, Kn1 = -5.0 + errors[2])

fodo_real = [qf_real, Drift(L=1.0), qd_real, Drift(L=1.0)]
beam_real = Beamline(fodo_real, R_ref=-59.52872449027632, species_ref=Species("electron"))

true_errors = errors[1:2]
model_kn1 = [5.0, -5.0]

initial_particle = [1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0]

function track_particle_mod()
    coords_i = copy(initial_particle)
    bunch = Bunch(coords_i, species=beam_mod.species_ref, R_ref=beam_mod.R_ref)
    track!(bunch, beam_mod)
    return [bunch.coords.v[1], bunch.coords.v[2], bunch.coords.v[3], bunch.coords.v[4]]
end

function track_particle_real()
    coords_i = copy(initial_particle)
    bunch = Bunch(coords_i, species=beam_real.species_ref, R_ref=beam_real.R_ref)
    track!(bunch, beam_real)
    return [bunch.coords.v[1], bunch.coords.v[2], bunch.coords.v[3], bunch.coords.v[4]]
end

function create_fodo_with_quads(kn1_array)
    qf = Quadrupole(L = 0.5, Kn1 = kn1_array[1])
    qd = Quadrupole(L = 0.5, Kn1 = kn1_array[2])
    fodo = [qf, Drift(L=1.0), qd, Drift(L=1.0)]
    return Beamline(fodo, R_ref=-59.52872449027632, species_ref=Species("electron"))
end

function track_particle(kn1_array)
    T = eltype(kn1_array)
    beamline = create_fodo_with_quads(kn1_array)
    coords_i = T.(initial_particle)
    bunch = Bunch(coords_i, species=beamline.species_ref, R_ref=beamline.R_ref)
    track!(bunch, beamline)
    return [bunch.coords.v[1], bunch.coords.v[2], bunch.coords.v[3], bunch.coords.v[4]]
end
""")

result_mod  = scibmad.track_particle_mod()
result_real = scibmad.track_particle_real()

full_mod  = [float(v) for v in result_mod]
full_real = [float(v) for v in result_real]

true_errors_np = np.array(scibmad.true_errors)
model_kn1_np   = np.array(scibmad.model_kn1)
real_kn1_np    = model_kn1_np + true_errors_np

dtype = torch.float64
lr = 1e-3
iterations = 10000

q1 = torch.tensor([model_kn1_np[0]], dtype=dtype, requires_grad=True)
q2 = torch.tensor([model_kn1_np[1]], dtype=dtype, requires_grad=True)

optimizer = torch.optim.Adam([q1, q2], lr=lr)

def loss_function(kn1_tensor):
    result = scibmad.track_particle(kn1_tensor)  # autograd path via newpatch
    loss = sum((result[i] - full_real[i]) ** 2 for i in range(4))
    return loss

for i in range(iterations):
    optimizer.zero_grad()
    kn1_array = torch.cat([q1, q2])
    loss = loss_function(kn1_array)
    loss.backward()
    optimizer.step()
    if i % 999 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6e}")

optimized_kn1    = torch.cat([q1, q2]).detach().numpy()
optimized_errors = optimized_kn1 - model_kn1_np

```

# FODO Quadrupole Error Optimization Results 

| Quad | Model | Real (erroneous) | Optimized |
|------|-------|-----------------|-----------|
| QF   | 5.0000 | 5.1293        | 5.1293    |
| QD   | -5.0000 | -5.0497      | -5.0497   |

| Quad | True Error | Optimized Error | Difference |
|------|-----------|----------------|------------|
| QF   | 0.129345  | 0.129345       | 0.000000   |
| QD   | -0.049661 | -0.049661      | 0.000000   |