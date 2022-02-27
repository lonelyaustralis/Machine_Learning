from utils.data import fashion_mnist_dataset, Dataloader
from Model.base_ulits import Sequential
from Model.Linear import Linear
from Model.Activation_func import Sigmoid, Relu, tanh, Sin, x_2, LL_Relu
from Model.Loss import BCE_loss_with_logit, CE_loss_with_softmax
import numpy as np
from utils.anime import Anime_plot
import matplotlib.pyplot as plt
from utils.learning_rate import warm_up_lr


def one_hot_label(label):
    label_one_hot = np.zeros((label.shape[0], 10))
    label = np.squeeze(label)
    label_one_hot[range(label.shape[0]), label] = 1
    return label_one_hot


train_dataset = fashion_mnist_dataset(path="./data/fashion", kind="train")
test_dataset = fashion_mnist_dataset(path="./data/fashion", kind="test")
train_dataloader = Dataloader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = Dataloader(test_dataset, batch_size=10000, shuffle=False)
epoch_num = 10

Linear_1 = Linear(784, 256, biases=True, normalization=False)
Linear_2 = Linear(256, 10, biases=True, normalization=False)
# Linear_3 = Linear(50, 3, biases=True, normalization=False)
# Linear_4 = Linear(3, 10, biases=True, normalization=False)
model = Sequential([Linear_1, Sigmoid(), Linear_2, CE_loss_with_softmax()])
# model = Sequential([Linear_1, Relu(), Linear_2, Relu(), Linear_3, Relu(), Linear_4, CE_loss_with_softmax()])
Animeor = Anime_plot(
    x_lim=[0, epoch_num],
    y_lim=[0, 1],
    data_x_start_point=[0] * 2,
    data_y_start_point=[0] * 2,
    legend=["loss", "test_acc"],
)
learning_rate = 0.5
for epoch in range(epoch_num):
    for index, (img, label) in enumerate(train_dataloader):
        lr = warm_up_lr(len(label), model.step, warm_up_steps=2000)
        model.train_one_time(img, label, lr)
    loss = np.mean(model.loss_history)
    print(loss)
    model.loss_history = []
    for index, (img, label) in enumerate(test_dataloader):
        _, _, acc, _, _ = model.eval(img, one_hot_label(label))
        acc = acc / 100
    Animeor.update([loss, acc])
Animeor.end()
