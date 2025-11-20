
# Scibmad

## What is scibmad?

Scibmad is a Julia-PyTorch binding that combines the convenience and accuracy of Julia-based automatic differentiation with the flexibility and ecosystem of PyTorch. It allows users to leverage Julia’s performance for scientific computing while working seamlessly in the PyTorch environment.

## Why is it useful?

- Provides faster and more precise gradient computations via Julia’s automatic differentiation.
- Integrates with PyTorch, making it easy for machine learning practitioners to use Julia features without leaving Python.
- Supports optimization workflows and complex scientific computations that benefit from Julia’s high-performance capabilities.

## Supported Platforms

- Python 3.10+
- Julia 1.9+
- macOS, Linux, Windows

## Installation

You can install scibmad via:

``` bash
pip install git+https://github.com/ChanpiseyU/scibmad.git
```

### Additional dependencies:

- JuliaCall (to interface with Julia):

``` bash
pip install JuliaCall
```

- PyTorch (for optimization routines):

``` bash
pip install torch 
```

- DifferentiationInterface (for Julia AD):
```bash
jl.seval('using Pkg; Pkg.add("DifferentiationInterface")')
```

- ForwardDiff (AD Backend):
```bash 
jl.seval('using Pkg; Pkg.add("ForwardDiff")')
```

## Getting Started
1. Install the project and dependencies (see Installation).
2. Import or define your Julia module via JuliaCall in Python.
3. Use Julia’s optimized functions in PyTorch workflows.


## Example

Minimizing Chebyshev Polynomial

``` python
# Define a Julia Chebyshev polynomial function
jl.seval("""
function chebyshev_t3(x)
    if x isa AbstractArray
        return 4 .* x.^3 .- 3 .* x
    else
        return 4 * x^3 - 3 * x
    end
end
""")

# Sample points
x2 = torch.linspace(-1, 1, 1000, dtype=torch.float64)
y2 = torch.cos(math.pi * x2)

# Parameters to optimize
a2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
b2 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
c2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.Adam([a2, b2, c2], lr=1e-3)

for i in range(500):
    optimizer.zero_grad()
    
    # Call the Julia function directly inside the computation graph
    y_pred = a2 + b2 * jl.chebyshev_t3(x2 + c2)
    
    loss = (y_pred - y2).pow(2).sum()
    
    loss.backward()
    optimizer.step()
```

For more detailed tutorials and examples, see the [User Guide](scibmad/user_guide.md).


## Troubleshooting / FAQ

- Ensure Julia is installed and in your system PATH.
- Import modules in the following order:

```python 

import scibmad 
from juliacall import Main as jl
import juliacall
import torch 

```

## References

- Julia [https://julialang.org/]
- JuliaCall [https://pypi.org/project/juliacall/]
- PyTorch [https://pytorch.org/]
- DifferentiationInterface [https://github.com/JuliaDiff/DifferentiationInterface.jl]
- ForwardDiff.jl [https://github.com/JuliaDiff/ForwardDiff.jl]

## License
This project is licensed under the MIT License. See [LICENSE](https://github.com/ChanpiseyU/scibmad/blob/main/LICENSE) for details.
 



