import nn.autodiff as ad

A = ad.Variable([[1], [2], [3]])

print(ad.grad(A * 2, [A]))