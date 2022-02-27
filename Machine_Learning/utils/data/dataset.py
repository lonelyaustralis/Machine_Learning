import sys, os

config_path = os.getcwd()
for i in range(len(__name__.split("."))):
    config_path = os.path.join(config_path, "../")
config_path = os.path.abspath(config_path)
sys.path.append(config_path)

from utils.mnist_utils.mnist_reader import load_mnist


class fashion_mnist_dataset:
    def __init__(self, path, kind="train"):
        if kind == "train":
            self.img, self.labels = load_mnist(path, kind="train")
        elif kind == "test":
            self.img, self.labels = load_mnist(path, kind="t10k")
        else:
            raise ValueError("kind must be 'train' or 'test'")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return self.img[idx] / 255, self.labels[idx]
