import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import  Variable
import matplotlib.pyplot as plt

# 准备测试数据
x_train = np.array([[3.4], [4.3], [5.8], [6.6], [6.9], [4.1], [9.8],
                   [6.2], [7.6], [2.15], [7.23], [10.2], [5.26], [8.0], [3.2], [10.2]], dtype=np.float32)

y_train = np.array([[1.6], [2.78], [2.02], [3.14], [1.7], [1.3], [3.6], [2.8], [2.4], [1.3],
                   [2.9], [3.6], [1.6], [2.8], [1.5], [3.7]], dtype=np.float32)
# x_train = np.array([3.4, 4.3, 5.8, 6.6, 6.9, 4.1, 9.8,
#                    6.2, 7.6, 2.15, 7.23, 10.2, 5.26, 8.0, 3.2, 10.2], dtype=np.float32)
#
# y_train = np.array([1.6, 2.78, 2.02, 3.14, 1.7, 1.3, 3.6, 2.8, 2.4, 1.3,
#                    2.9, 3.6, 1.6, 2.8, 1.5, 3.7], dtype=np.float32)
assert  x_train.size == y_train.size
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 设计一元线性模型

class OneLinearRegression(nn.Module):
    def __init__(self):
        super(OneLinearRegression, self).__init__()
        self.regression = nn.Linear(1,1)

    def forward(self, x):
        out = self.regression(x)
        return out

# 创建一元线性模型的一个实例

model = OneLinearRegression()

# 怎么判断它的准则,损失函数吧

Discriminant_rule = nn.MSELoss()

# 优化方法
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epoch  = 2000
for i in range(epoch):
    x_train = Variable(x_train)
    y_train = Variable(y_train)

    ## 获取模型输出值
    out = model(x_train)

    # 损失函数值
    loss = Discriminant_rule(y_train, out)

    # 清空参数的所有梯度
    optimizer.zero_grad()

    # 计算梯度
    loss.backward()
    # 跟新参数
    optimizer.step()

    if(i % 100 == 0 ):
        print('epoch {}, loss {}'.format(epoch, loss.item()))

x_train = Variable(x_train)

model.eval()
predict = model(x_train)
predict = predict.data.numpy()
plt.plot(x_train.data.numpy, y_train.data.numpy(), 'go', label='Original data')
plt.plot(x_train.data.numpy, predict, label='FittingLine')
# plt.plot(range(x_train.data.numpy), range(y_train.data.numpy()), 'ro', 'Original data')
# plt.plot(range(x_train.data.numpy), range(predict), label = 'fittingLine')
plt.show()


