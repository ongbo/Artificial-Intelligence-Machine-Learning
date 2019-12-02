import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
print(x_train)
print(y_train)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    out = model(inputs) # 将inputs的值送给model，然后计算出预测值
    loss = criterion(out, target)#将out和目标值target比较误差

    optimizer.zero_grad() # 清空上一步的参与更新参数值
    loss.backward() # 误差反向传播， 计算参数更新值
    optimizer.step() # 将参数更新施加到model的parameters
    if epoch % 20 == 0:
        plt.cla()
        plt.scatter(x_train.numpy(),y_train.numpy())
        plt.plot(x_train.numpy(),out.data.numpy(),'r-',lw=5)
        plt.pause(0.1)

# plt.figure()
# plt.scatter(x_train.numpy(),y_train.numpy())

# plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
# plt.plot(x_train.numpy(),)
plt.show()