from .base_ulits import base_model
import numpy as np
from typing import Union


def sigmoid_func(inputs: np.ndarray):
    """
    为了防止当x小于0时，绝对值太大导致e^-x会溢出的情况，我们对x大于0和小于0的情况分别处理
    当x大于0时采用sigmoid的原始样式，z = 1 / (1 + e^-x)
    当x小于0时，对原始公式上下同时乘以e^x次方，z = e^x / (1 + e^x)

    Parameters
    ----------
    x : [type:numpy array]
        [description]
        进行sigmoid计算的数据
    Returns
    -------
        [type:numpy array]
        [description]
        返回sigmoid计算后的结果
    """
    # mask是一个布尔型矩阵，x大于0的位置为True，x小于0的位置为False
    mask = inputs > 0
    # 生成一个大小为x的矩阵，元素全部为0，分别用于存放x大于0时的结果和x小于0时的结果
    positive_result = np.zeros_like(inputs, dtype=np.float64)
    negative_result = np.zeros_like(inputs, dtype=np.float64)
    # 调用np.exp函数计算e^-x，where可以指定运算x中的那些位置，mask中为True的就进行运算，为False的不运算，结果存放到positive_result中
    np.exp(-inputs, out=positive_result, where=mask)
    # 调用np.exp函数计算e^x，where可以指定运算x中的那些位置，~mask为mask的取反，故而mask中为False的就进行运算，为True的不运算，结果存放到negative_result中
    np.exp(inputs, out=negative_result, where=~mask)
    # 计算z = 1 / (1 + e^-x)
    positive_result = 1 / (1 + positive_result)
    # 上述公式会使得mask为False的位置为1，因为其对应位置的元素值为0，所以这里使用~mask作为数组的索引，将其对应位置的元素值设置为0
    positive_result[~mask] = 0
    # 计算z = e^x / (1 + e^x)
    negative_result = negative_result / (1 + negative_result)
    # 上述公式会使得mask为True的位置为1，因为其对应位置的元素值为0，所以这里使用mask作为数组的索引，将其对应位置的元素值设置为0
    negative_result[mask] = 0
    # 将计算结果相加，即得到z
    result = positive_result + negative_result
    return result


class Sigmoid(base_model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.forward_result = sigmoid_func(inputs)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        return np.multiply(self.forward_result * (1 - self.forward_result), top_gard)

    def update(self, *args):
        pass


class tanh(base_model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.forward_result = np.tanh(inputs)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        return np.multiply(1 - np.power(self.forward_result, 2), top_gard)

    def update(self, *args):
        pass


class Relu(base_model):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.dec_mat = (inputs > 0).astype(np.float64)
        self.forward_result = np.multiply(inputs, self.dec_mat)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        return np.multiply(top_gard, self.dec_mat)

    def update(self, *args):
        pass


class Sin(base_model):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        self.forward_result = np.sin(inputs)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        return np.multiply(np.cos(self.inputs), top_gard)

    def update(self, *args):
        pass


class x_2(base_model):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        self.forward_result = np.power(inputs, 2)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        return np.multiply(2 * (self.inputs), top_gard)

    def update(self, *args):
        pass


class LL_Relu(base_model):
    def __init__(self) -> None:
        super().__init__()
        self.__have_params = True
        self.k = 1
        self.a = 0.01

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        self.dec_mat = (inputs > 0).astype(np.float64)
        self.dec_mat[self.dec_mat == 0] = self.a
        self.dec_mat[self.dec_mat == 1] = self.k
        self.forward_result = np.multiply(inputs, self.dec_mat)
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        self.gard = np.multiply(top_gard, self.dec_mat)
        k_parms = np.multiply(self.inputs[self.dec_mat == self.k], top_gard[self.dec_mat == self.k])
        a_parms = np.multiply(self.inputs[self.dec_mat == self.a], top_gard[self.dec_mat == self.a])
        self.k_gard = np.sum(k_parms) / (self.inputs.shape[0] * self.inputs.shape[1])
        self.a_gard = np.sum(a_parms) / (self.inputs.shape[0] * self.inputs.shape[1])
        return self.gard

    def update(self, learning_rate: float):
        # learning_rate = learning_rate * 0.5
        self.k = self.k - self.k_gard * learning_rate
        # learning_rate = learning_rate * 0.01
        self.a = self.a - self.a_gard * learning_rate


def Activation_func_gard_check(func: base_model, value: np.ndarray, deviation=1e-4):
    value = np.array(value)
    func(value)
    func_gard = func.backward(np.array([1]))
    func_value_plus = func(value + deviation)
    func_value_minus = func(value - deviation)
    func_gard_numic = (func_value_plus - func_value_minus) / (2 * deviation)
    return abs(func_gard_numic - func_gard), abs(func_gard_numic / (func_gard + func_gard_numic))


class erf(base_model):
    # 该函数使用数值近似法计算，计算精度为0.00001，算法详细查询 https://zh.wikipedia.org/wiki/%E8%AF%AF%E5%B7%AE%E5%87%BD%E6%95%B0
    def __init__(self):
        super().__init__()
        self.p = 0.3275911
        self.a1 = 0.254829592
        self.a2 = -0.284496736
        self.a3 = 1.421413741
        self.a4 = -1.453152027
        self.a5 = 1.061405429
        # self.a = 0.140012

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.dec_mat = (inputs > 0).astype(np.float64)
        self.dec_mat[self.dec_mat == 0] = -1
        self.inputs = np.abs(inputs)
        t = 1 / (1 + self.p * self.inputs)
        self.forward_result = 1 - (
            self.a1 * t + self.a2 * np.power(t, 2) + self.a3 * np.power(t, 3) + self.a4 * np.power(t, 4) + self.a5 * np.power(t, 5)
        ) * np.exp(-np.power(self.inputs, 2))
        return np.multiply(self.forward_result, self.dec_mat)

        # x_2 = np.power(inputs, 2)
        # result = np.sign(inputs) * (1 - np.exp(-x_2 * (4 / np.math.pi + self.a * x_2) / (1 + self.a * x_2)))
        # return result


class GELU(base_model):
    def __init__(self):
        super().__init__()
        self.erf_func = erf()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        self.erf_value = self.erf_func(self.inputs / np.sqrt(2.0))
        self.forward_result = 0.5 * (1 + self.erf_value)
        return np.multiply(self.forward_result, self.inputs)

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        self.grad = 0.5 * (1 + self.erf_value + self.inputs * np.sqrt(2 / np.math.pi) * np.exp(-np.power(self.inputs, 2)))
        return np.multiply(self.grad, top_gard)
