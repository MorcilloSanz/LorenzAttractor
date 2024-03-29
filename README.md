# LorenzAttractor :butterfly:

The Lorenz attractor is a strange attractor living in 3D space that relates three parameters arising in fluid dynamics. It is one of the Chaos theory's most iconic images and illustrates the phenomenon now known as the Butterfly effect or (more technically) sensitive dependence on initial conditions

The model is a system of three ordinary differential equations now known as the Lorenz equations:

$\frac{dx}{dt} = \sigma(y - x)$

$\frac{dy}{dt} = x(\rho - z) - y$

$\frac{dz}{dt} = xy - \beta z$

One normally assumes that the parameters $\sigma$, $\rho$ and $\beta$ are positive. Lorenz used the values $\sigma = 10$, $\beta = 8/3$ and $\rho = 28$. The system exhibits chaotic behavior for these (and nearby) values

[Lorenz Attractor](https://mathworld.wolfram.com/LorenzAttractor.html)
[Lorenz System](https://en.wikipedia.org/wiki/Lorenz_system)

![alt text](https://github.com/MorcilloSanz/LorenzAttractor/blob/main/img/img.png)

## Dependencies

```
glfw
glew
glm
imgui
```
