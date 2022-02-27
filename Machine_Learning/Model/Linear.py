import numpy as np
from .Activation_func import sigmoid_func
from .base_ulits import base_model


class Linear_Simple(base_model):
    def __init__(self, n_features: int, biases: bool = True) -> None:
        super().__init__()
        w = np.zeros((n_features, 1))
        self.w = w
        self.biases = biases
        if biases:
            self.b = 0

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        assert inputs.shape[-1] == len(self.w), "Error: input size not match"
        if self.biases:
            self.forward_result = np.dot(inputs, self.w) + self.b
        else:
            self.forward_result = np.dot(inputs, self.w)
        return self.forward_result

    def backward(self, y: np.ndarray) -> None:
        diff = self.forward_result - y
        self.w_gard = np.dot(self.inputs.T, diff) / len(self.inputs)
        if self.biases:
            self.b_gard = np.sum(diff) / len(self.inputs)

    def zero_gard(self) -> None:
        self.w_gard = np.zeros_like(self.w)
        if self.biases:
            self.b_gard = 0

    def MSE_loss(self, y: np.ndarray) -> float:
        return np.mean(np.power((self.forward_result - y), 2))

    def update(self, learning_rate: float) -> None:
        self.w = self.w - learning_rate * self.w_gard
        if self.biases:
            self.b = self.b - learning_rate * self.b_gard

    def save_weight(self, filename: str) -> None:
        if self.biases:
            np.savez(filename, w=self.w, b=self.b)
        else:
            np.savez(filename, w=self.w)
        print("save_complete")

    def load_wegiht(self, filename: str) -> None:
        data = np.load(filename)
        self.w = data["w"]
        if self.biases:
            self.b = data["b"]


class Linear_Logit(Linear_Simple):
    def __init__(self, n_features: int, biases: bool = True, normalization: bool = True, get_higher_order: bool = False, order_num: int = 2) -> None:
        self.order_num = order_num
        self.get_higher_order = get_higher_order
        if self.get_higher_order:
            n_features = (self.order_num + 1) * (self.order_num + 2) // 2 - 1
        super().__init__(n_features, biases=biases)
        self.normalization = normalization
        self.train = True

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        if self.get_higher_order and self.train:
            self.inputs = self.higher_order(inputs, self.order_num)
        if self.normalization and self.train:
            self.mean = np.mean(self.inputs, axis=0)
            self.std = np.std(self.inputs, axis=0)
            self.inputs = (self.inputs - self.mean) / self.std
        self.forward_result = super().forward(self.inputs)
        self.forward_result = sigmoid_func(self.forward_result)
        return self.forward_result

    def inference(self, inputs: np.ndarray) -> np.ndarray:
        if self.get_higher_order:
            inputs = self.higher_order(inputs, self.order_num)
        if self.normalization:
            inputs = (inputs - self.mean) / self.std
        self.train = False
        return self.forward(inputs)

    def CE_Loss(self, y):
        loss = -np.mean(np.multiply(y, np.log(self.forward_result)) + np.multiply((1 - y), np.log(1 - self.forward_result)))
        return loss

    def higher_order(self, x: np.ndarray, order_num: int) -> np.ndarray:
        assert order_num >= 2, "Error: order_num must be >= 2"
        assert x.ndim == 2, "Error: x must be 2-D array"
        order_num = order_num + 1
        x1 = x[:, 0]
        x2 = x[:, 1]
        for i in range(0, order_num):
            for j in range(0, order_num - i):
                if (i == 0 and j == 0) or (i == 1 and j == 0) or (i == 0 and j == 1):
                    continue
                x = np.column_stack((x, np.multiply(np.power(x1, i), np.power(x2, j))))
        return x


class Linear(base_model):
    def __init__(self, channels_in: int, channels_out: int, Name: str = None, biases: bool = True, normalization: bool = False) -> None:
        super().__init__()
        self.Name = Name
        self.__have_params = True
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalization = normalization
        k = np.sqrt(1 / channels_in)
        # k = np.sqrt(6 / (channels_in + channels_out))
        self.weights = np.random.uniform(-k, k, size=(channels_out, channels_in))
        self.biases = biases
        if biases:
            self.bias = np.random.uniform(-k, k, size=(1, channels_out))

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        # forward_result dim: (batch_size, channels_out)
        self.forward_result = np.dot(inputs, self.weights.T)
        if self.biases:
            self.forward_result += self.bias
        return self.forward_result

    def backward(self, top_gard: np.ndarray) -> np.ndarray:
        # top_gard dim: (batch_size, channels_out)
        self.bottom_gard = np.dot(top_gard, self.weights)
        self.weights_gard = np.dot(top_gard.T, self.inputs) / top_gard.shape[0]
        if self.biases:
            self.bias_gard = np.sum(top_gard, axis=0, keepdims=True) / top_gard.shape[0]
        return self.bottom_gard

    def update(self, learning_rate: float) -> None:
        self.weights = self.weights - learning_rate * self.weights_gard
        if self.biases:
            self.bias = self.bias - learning_rate * self.bias_gard
