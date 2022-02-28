import pandas as pd
import matplotlib.pyplot as plt
from Model.Linear import Linear_Simple

# ---------------------------------Load Data---------------------------------
path = "data/simple_dataset/ex1data1.txt"
# ex1data2.txt需要进行特征缩放处理，否则需要使用极小的学习率，特征缩放实现见逻辑回归例子
data = pd.read_csv(path, header=None)
cols = data.shape[1]  # 获取数据的列数
X = data.iloc[:, 0 : cols - 1]  # 获取除了最后一列外的数据
y = data.iloc[:, cols - 1 : cols]  # 获取最后一列的数据
X = X.values
y = y.values
# ---------------------------------inital the model---------------------------------
My_Linear = Linear_Simple(n_features=2, biases=True)

iters = 1000  # iteration times
lr = 0.001  # learning rate
cost = []  # A list to store the cost
for i in range(iters):
    # ---------------------------------forward---------------------------------
    My_Linear(X)
    # ---------------------------------backward---------------------------------
    My_Linear.backward(y)
    # ---------------------------------update---------------------------------
    My_Linear.update(lr)
    # ---------------------------------cost---------------------------------
    cost.append(My_Linear.MSE_loss(y))
# ----------------------------plot the cost-----------------------------------
plt.figure(0)
plt.plot(cost)
plt.show()
