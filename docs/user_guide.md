
## Legendre Polynomial

``` python
jl.seval("""
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

    y_pred = a + b * jl.legendre_p3(c + d * x) 
    
    loss = (y_pred - y).pow(2).sum()
    
    loss.backward()
    optimizer.step()

```

## Bessel-Like Function Approximation

``` python 
jl.seval("""
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
    
    y_pred = jl.bessel_approx(scale * x3 + shift)
    
    loss = (y_pred - y3).pow(2).sum()
    
    loss.backward()
    optimizer1.step()

```
