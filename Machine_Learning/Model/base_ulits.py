import numpy as np
import json
import os


class base_model:
    def __init__(self):
        self.__have_params = False

    def get_params_state(self) -> bool:
        return self.__have_params

    def set_params_state(self, state: bool) -> None:
        self.__have_params = state

    def __call__(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        return self.forward(inputs, **kwargs)

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        pass


def Sequential(layers: list):
    return Module(layers)


class Module(base_model):
    def __init__(self, model_list: list, get_loss_history: bool = False):
        self.model_list = model_list
        self.get_loss_history = get_loss_history
        self.loss_history: list = []
        self.step = 1

    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        self.inputs = inputs
        for index, model in enumerate(self.model_list):
            if index == len(self.model_list) - 1:
                break
            inputs = model(inputs, **kwargs)

        self.forward_result = inputs
        return self.forward_result

    def loss(self, labels: np.ndarray) -> float:
        loss = self.model_list[-1](inputs=self.forward_result, labels=labels)
        self.loss_history.append(loss)
        return loss

    def backward(self) -> None:
        for index, model in enumerate(reversed(self.model_list)):
            if index == 0:
                top_gard = model.backward()
            else:
                top_gard = model.backward(top_gard)

    def update(self, learning_rate: float) -> None:
        for model in self.model_list:
            if model.get_params_state:
                model.update(learning_rate)

    def recall(self, eval_result: np.ndarray, labels: np.ndarray, unique_label: np.ndarray) -> dict:
        recall_result = {}
        for label_id in unique_label:
            correct_result = np.logical_and(labels == label_id, eval_result == label_id)
            correct_result = correct_result.astype(np.int).sum()
            recall_result[label_id] = (correct_result / np.sum(labels == label_id)) * 100
        return recall_result

    def precision(self, eval_result: np.ndarray, labels: np.ndarray, unique_label: np.ndarray) -> dict:
        precision_result = {}
        for label_id in unique_label:
            correct_result = np.logical_and(labels == label_id, eval_result == label_id)
            correct_result = correct_result.astype(np.int).sum()
            precision_result[label_id] = (correct_result / np.sum(eval_result == label_id)) * 100
        return precision_result

    def accuracy(self, eval_result: np.ndarray, labels: np.ndarray) -> float:
        return ((eval_result == labels).astype(np.int).sum() / labels.size) * 100

    def eval(self, inputs: np.ndarray, one_hot_label: np.ndarray) -> np.ndarray:
        labels = np.argmax(one_hot_label, axis=1)
        eval_result = np.argmax(self.forward(inputs), axis=1)
        unique_label = np.unique(labels)
        recall_result = self.recall(eval_result, labels, unique_label)
        precision_result = self.precision(eval_result, labels, unique_label)
        accuracy_result = self.accuracy(eval_result, labels)
        avg_recall = np.mean(list(recall_result.values()))
        avg_precision = np.mean(list(precision_result.values()))
        print("-" * 9 + "-" * 100)
        print(" " * 9 + " " * 45 + "eval_result")
        print("-" * 9 + "-" * 100)
        space_num = (100 - len(unique_label)) // len(unique_label)
        print("label_id", end="  |")
        for label_id in unique_label:
            print(str(label_id), end=" " * space_num)
        print()
        print("recall", end="    |")
        for label_id in unique_label:
            print_word = "{:.3f}%".format(recall_result[label_id])
            id_sapce_num = space_num - len(print_word) + 1
            print(print_word, end=" " * id_sapce_num)
        print()
        print("precision", end=" |")
        for label_id in unique_label:
            print_word = "{:.3f}%".format(precision_result[label_id])
            id_sapce_num = space_num - len(print_word) + 1
            print(print_word, end=" " * id_sapce_num)
        print()
        print("accuracy", end="  |")
        print("{:.3f}%".format(accuracy_result))
        print("avg_recall", end="|")
        print("{:.3f}%".format(avg_recall))
        print("avg_precis", end="|")
        print("{:.3f}%".format(avg_precision))
        return recall_result, precision_result, accuracy_result, avg_recall, avg_precision

    def train_one_time(self, inputs, lables, learning_rate):
        self.forward(inputs)
        self.loss(lables)
        self.backward()
        self.update(learning_rate)
        self.step += 1
        return self.loss_history[-1]

    def save_weights(self, path: str, file_name) -> None:
        json_list = []
        for index, model in enumerate(self.model_list):
            layer_info = {"class_name": model.__class__.__name__ + str(index), "params": model.get_params_state()}
            if model.get_params_state():
                data = model.get_params()
                npz_file_name = file_name + "_" + model.__class__.__name__ + str(index) + ".npz"
                np.savez(os.path.join(path, npz_file_name), **data)
            json_list.append(layer_info)
        with open(os.path.join(path, file_name + ".json"), "w") as f:
            json.dump(json_list, f)

    def load_weights(self, path: str, file_name) -> None:
        with open(os.path.join(path, file_name + ".json"), "r") as f:
            json_list = json.load(f)
        for index, layer_info in enumerate(json_list):
            class_name = layer_info["class_name"]
            if layer_info["params"]:
                data_array_dict = np.load(os.path.join(path, file_name + "_" + class_name + ".npz"))
                self.model_list[index].set_params(data_array_dict)

