from Model.base_ulits import Sequential
from Model.Linear import Linear
from Model.Activation_func import Sigmoid, Relu, tanh, Sin, x_2, LL_Relu
from Model.Loss import BCE_loss_with_logit, CE_loss_with_softmax
import numpy as np
import scipy.io as sio
from utils.anime import Anime_plot
import matplotlib.pyplot as plt


def dataloader(filename):
    data_mat = sio.loadmat(filename)
    img = data_mat["X"]
    label = data_mat["y"]
    label[label == 10] = 0
    return img, label


def one_hot_label(label):
    label_one_hot = np.zeros((label.shape[0], 10))
    label = np.squeeze(label)
    label_one_hot[range(label.shape[0]), label] = 1
    return label_one_hot


def div_dataset(inputs, labels, div_rate):
    idx = np.random.choice(inputs.shape[0], size=int(inputs.shape[0] * div_rate), replace=False)
    train_inputs = inputs[idx]
    train_labels = labels[idx]
    test_inputs = np.delete(inputs, idx, axis=0)
    test_labels = np.delete(labels, idx, axis=0)
    for i in range(0, 10):
        print(sum(test_labels == i))
    return train_inputs, train_labels, test_inputs, test_labels


if __name__ == "__main__":
    Linear_1 = Linear(400, 25, biases=True, normalization=False)
    Linear_2 = Linear(25, 10, biases=True, normalization=False)
    model = Sequential([Linear_1, Relu(), Linear_2, CE_loss_with_softmax()])
    img, label = dataloader("./data/simple_dataset/ex4data1.mat")
    train_img, train_labels, test_img, test_labels = div_dataset(img, label, 0.6)
    train_one_hot_label = one_hot_label(train_labels)
    test_one_hot_label = one_hot_label(test_labels)
    iters = 2000
    learning_rate = 0.3
    eval_rate = 50
    the_anime_plot = Anime_plot(
        x_lim=[-5, iters // eval_rate + 10],
        y_lim=[0, 105],
        data_x_start_point=[0] * 6,
        data_y_start_point=[0] * 6,
        legend=["train_acc", "test_acc", "train_avg_recall", "test_avg_recall", "train_avg_precision", "test_avg_precision"],
    )
    for i in range(iters):
        model.train_one_time(train_img, train_labels, learning_rate)
        if (i + 1) % eval_rate == 0:
            print(model.loss_history[-1])
            print("Iteration: {}, train_data_eval".format(i + 1))
            _, _, train_acc, train_avg_reacll, train_avg_precis = model.eval(train_img, train_one_hot_label)
            print("Iteration: {}, test_data_eval".format(i + 1))
            _, _, test_acc, test_avg_reacll, test_avg_precis = model.eval(test_img, test_one_hot_label)
            the_anime_plot.update(data_y_update=[train_acc, test_acc, train_avg_reacll, test_avg_reacll, train_avg_precis, test_avg_precis])
    the_anime_plot.end()
    print("done")
