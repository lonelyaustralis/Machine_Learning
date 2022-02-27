import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model.Linear import Linear_Logit


# ---------------------------------Load Data---------------------------------
# You can chose ex2data1.txt or ex2data2.txt or you own dataset
path = "data/simple_dataset/ex2data2.txt"
data = pd.read_csv(path, header=None)
cols = data.shape[1]  # 获取数据的列数
X = data.iloc[:, 0 : cols - 1]  # 获取除了最后一列外的数据
y = data.iloc[:, cols - 1 : cols]  # 获取最后一列的数据
X = X.values
y = y.values
# ---------------------------------inital the model---------------------------------
my_linear = Linear_Logit(n_features=2, biases=False, normalization=True, get_higher_order=True, order_num=3)

iters = 1000  # iteration times
lr = 0.5  # learning rate
cost = []  # A list to store the cost
for i in range(iters):
    # ---------------------------------forward---------------------------------
    my_linear(X)
    # ---------------------------------backward---------------------------------
    my_linear.backward(y)
    # ---------------------------------update---------------------------------
    my_linear.update(lr)
    # ---------------------------------cost---------------------------------
    cost.append(my_linear.CE_Loss(y))

# ----------------------------plot the cost-----------------------------------
plt.figure(0)
plt.plot(cost)

# ---------------------plot the decision boundary----------------------------------
plt.figure(1)
positive = X[(y == 1).squeeze(1)]
negative = X[(y == 0).squeeze(1)]
plt.scatter(positive[:, 0], positive[:, 1], s=50, c="b", marker="o", label="positive")
plt.scatter(negative[:, 0], negative[:, 1], s=50, c="r", marker="x", label="negative")
mesh_x = np.linspace(X[:, 0].min() - X[:, 0].min() * 0.1, X[:, 0].max() + X[:, 0].max() * 0.1, 1000)
mesh_y = np.linspace(X[:, 1].min() - X[:, 1].min() * 0.1, X[:, 1].max() + X[:, 1].max() * 0.1, 1000)
mesh_x, mesh_y = np.meshgrid(mesh_x, mesh_y)
mesh_z = my_linear.inference(np.column_stack((mesh_x.ravel(), mesh_y.ravel()))).reshape(mesh_x.shape)
plt.contour(mesh_x, mesh_y, mesh_z, levels=[0.3, 0.5, 0.7, 0.9], colors=["red", "green", "blue", "yellow"], linestyles=["--", "-.", ":", "-"])


# ---------------------------------eval the accuracy of the model---------------------------------
threshold = 0.5
result = my_linear.inference(X)
result = result > threshold
acc = np.logical_and(y == 1, result).sum() / y.shape[0]
print("accuracy:", acc)
plt.show()
print("done")
