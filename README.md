
# Scibmad

## What is scibmad?

Scibmad is a Julia-PyTorch binding that combines the convenience and accuracy of Julia-based automatic differentiation with the flexibility and ecosystem of PyTorch. 

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


## Examples

For detailed examples on using scibmad for torch optimization, see the [User Guide](scibmad/user_guide.md).


## Troubleshooting / FAQ

- Ensure `scibmad.seval()` is used to define Julia functions
- If gradients aren't flowing, ensure `requires_grad=True` is set on your tensors
- Julia packages must be installed in your Julia environment before calling `using PackageName`
- Import modules in the following order:

```python 

from scibmad import core as scibmad
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
 



