## Machine Learning


## Update 2022-3-10
更新了GELU激活函数，以及一个用于检查激活函数的反向梯度是否计算正确的工具函数，均放在Model/Activation_func.py文件中。工具函数的样例见GELU_check.py

## 简介

一份用纯numpy实现的机器学习代码，包含线性回归，逻辑回归，以及全连接神经网络三个部分

**线性回归示例**：Linear_Regression.py 使用数据集为吴恩达机器学习课程所提供的两个数据集：ex1data1与ex1data2

**逻辑回归示例**：Logit_Regression.py 使用数据集为吴恩达机器学习课程所提供的两个数据集：ex2data2与ex2data2

**全连接神经网络示例**：Small_MNIST_NN_example.py 使用数据集为吴恩达机器学习课程提供的小型MNIST数据集，供5000张手写数字图片，Fashion_MNIST_example.py  使用完整Fashion_MNIST数据集进行训练，Fashion_MNIST数据集下载地址：https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

下载完后将所有文件放入./data/fashion文件夹下

 环境要求：
 
 Python 3.8
 
 numpy==1.20.3
 
 matplotlib==3.4.3
 
 pandas==1.3.4
