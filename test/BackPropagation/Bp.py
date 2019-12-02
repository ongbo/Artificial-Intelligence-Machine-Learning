# import numpy as np
# import torch.nn as nn
# #先定义训练数据
# train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
# train_Y = np.array([0,1,1,0])
#
# #线性连接层
# class LinearLayer:
#     def __init__(self, input_D, output_D):
#         self._input_D, output_D))W = np.random.normal(0, 0.1, (
#         self._b = np.random.normal(0, 0.1, (1,output_D))
#         self._grad_W = np.zeros((input_D,output_D))
#         self._grad_b = np.zeros((1,output_D))
#
#     def forward(self, X):
#         return np.matmul(X, self._W) + self._b
#
#     def backward(self, X, grad):
#         self._grad_W = np.matmul(X.T, grad)
#         self._grad_b = np.matmul(grad.T, np.ones(X.shape[0]))
#         return np.matmul(grad, self._W.T)
#
#     def update(self, learn_rate):
#         self._W = self._W - self._grad_W * learn_rate
#         self._b = self._b - self._grad_b * learn_rate
#
# class P:
#     def __init__(self):
#         pass
#     def forward(self, X):
#         return np.where(X < 0, 0, X)
#     def backward(self, X, grad):
#         return np.where(X>0, X, 0) * grad
#
#
# linear1 = LinearLayer(2,3)
# p = P()
# linear2 = LinearLayer(3,1)
# model = [linear1, p, linear2]
# def predict(model, X):
#     tmp = X
#     for layer in model:
#         tmp = layer.forward(tmp)
#     return np.where(tmp > 0.5, 1, 0)
#
# print("----------")
# result = predict(model, train_X)
#
# #训练网络
# def MSEloss(train_Y, y):
#     assert train_Y.shape == y.shape
#     return np.linalg.norm(train_Y - y) ** 2
#
#
# for i in range(50000):
#
#     #先来前向传播，获取每个值
#     o0 = train_X
#     a1 = linear1.forward(o0)
#     o1 = p.forward(a1)
#     a2 = linear2.forward(o1)
#     o2 = a2
#
#     #根据前馈算法，来计算loss
#     y = o2.reshape(o2.shape[0])
#     loss = MSEloss(train_Y, y)
#
#     #反向传播更新参数
#     grad = (y-train_Y).reshape(result.shape[0], 1)
#     grad = linear2.backward(o1,grad)
#     grad = p.backward(a1, grad)
#     grad = linear1.backward(o0, grad)
#
#     learn_rate = 0.001
#
#     #更新参数
#     linear1.update(learn_rate)
#     linear2.update(learn_rate)
#
#     #判断学习是否完成
#     if i% 200 == 0:
#         print(loss)
#     if loss < 0.001:
#         print("训练完成")
#         break
#
# print("预测数据1")
# print(train_X)
# print("预测结果")
# print(result)
#
