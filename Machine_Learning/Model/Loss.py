import numpy as np
from .base_ulits import base_model
from .Activation_func import sigmoid_func
import time


class BCE_loss_with_logit(base_model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = sigmoid_func(inputs)
        self.labels = kwargs["labels"]
        loss = -np.sum(np.multiply(self.labels, np.log(self.inputs)) + np.multiply((1 - self.labels), np.log(1 - self.inputs))) / self.inputs.shape[0]
        return loss

    def backward(self) -> np.ndarray:
        self.grad = self.inputs - self.labels
        return self.grad

    def update(self, *args):
        pass


class CE_loss_with_softmax(base_model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = np.exp(inputs)
        self.labels = kwargs["labels"]
        if len(self.labels.shape) != 1:
            self.labels = self.labels.squeeze(1)
        self.inputs_sum = np.sum(self.inputs, axis=1)
        self.forward_result = self.inputs[range(self.labels.shape[0]), self.labels] / self.inputs_sum
        loss = -np.sum(np.log(self.forward_result)) / self.inputs.shape[0]
        return loss

    def backward(self) -> np.ndarray:
        self.gard = self.inputs / self.inputs_sum.reshape(self.inputs.shape[0], 1)
        self.gard[range(self.labels.shape[0]), self.labels] -= 1
        return self.gard

    def update(self, *args):
        pass
