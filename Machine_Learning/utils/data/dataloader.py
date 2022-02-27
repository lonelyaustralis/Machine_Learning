import random


class Dataloader:
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iteration = 0
        self.epoch = 0
        self.idx_list = list(range(len(self.dataset)))

    def __iter__(self):
        self.start_idx = 0
        if self.shuffle:
            random.shuffle(self.idx_list)
        return self

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            self.epoch += 1
            raise StopIteration
        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_idx = self.idx_list[self.start_idx : end_idx]
        self.start_idx = end_idx
        self.iteration += 1
        return self.dataset[batch_idx]