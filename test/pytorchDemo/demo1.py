# import print_function from __future__
import torch
x = torch.Tensor(5,3) #构造一个5*3的矩阵
x = torch.rand(5,3) #随机构造一个5*3的矩阵

print(x)
print(x.size())

y = torch.rand(5,3)
print(y)
x = torch.add(x,y)
print(x)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x = x+y
    print(x)
else:
    print("is not available")
