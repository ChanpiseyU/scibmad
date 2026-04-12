# Scibmad – PyTorch/Julia Binding 

## Overview

This repository provides the **PyTorch ↔ Julia automatic differentiation bridge** used within the broader Scibmad ecosystem.

It enables seamless integration of:
- Julia-based scientific computing and AD (via ForwardDiff / DifferentiationInterface)
- PyTorch’s optimization and machine learning workflows

---

## Role in the Scibmad Ecosystem

Within the full Scibmad stack, this module is responsible for:

- Bridging Python tensors ↔ Julia arrays  
- Enabling PyTorch autograd compatibility with Julia functions  
- Handling forward and backward passes across language boundaries  
- Providing differentiable wrappers for Julia-based physics routines  

---

## Requirements

- Python 3.10+  
- Julia 1.9+  
- macOS, Linux, or Windows  

---

## Installation

This module is typically installed as part of the main Scibmad package.

For development or standalone testing:

```bash
pip install git+https://github.com/ChanpiseyU/scibmad.git
```

## Additional Dependencies 

- PyTorch 
``` bash 
pip install torch
```

- Julia AD dependencies 
``` bash
jl.seval('using Pkg; Pkg.add("DifferentiationInterface")')
jl.seval('using Pkg; Pkg.add("ForwardDiff")')
```

## Usage Context 

Typical usage patterns:
``` python 
from scibmad import core as scibmad
import torch
```

Julia functions are defined byL
```python 
scibmad.define("function f(x); return x^2; end")
```
and can then be used directly in PyTorch optimization loops with autograd support.

## Key features 

- Cross-language automatic differentiation 
- PyTorch-compatible autograd interface
- Transparent tensor ↔ Julia type conversion
- Support for scalar and array-valued functions
- Designed for high-performance scientific computing workflows

## Notes 

- Import order matters:
``` python
from scibmad import core as scibmad
import torch
```

- Ensure:
    - requires_grad=True is set for optimization
    - Julia packages are installed before use
    - Functions are defined in Julia before calling them

## References: 
- [Julia](https://julialang.org/)
- [JuliaCall](https://pypi.org/project/juliacall/)
- [PyTorch](https://pytorch.org/)
- [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
- [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)

## License 
MIT License — see the main Scibmad repository for details.