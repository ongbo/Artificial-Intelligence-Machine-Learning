import torch
import numpy as np
x = torch.rand(5,3)

print(x)

print(x[1:3,2])

#numpy和pytorch公用
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)


