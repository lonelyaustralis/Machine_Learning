import numpy as np
import cv2


class rotation_img:
    def __init__(self, angle_range: list = [-10, 10], random_rate: float = 0.5):
        assert len(angle_range) == 2, "angle_range must be a list with 2 elements"
        self.angle_range = angle_range
        self.random_rate = random_rate

    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray), "img must be a numpy array"
        if np.random.rand() < self.random_rate:
            angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            rows = img.shape[0]
            cols = img.shape[1]
            rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
            img = cv2.warpAffine(img, rotate, (cols, rows))
        return img


class Horizontal_rotation:
    def __init__(self, random_rate: float = 0.5):
        self.random_rate = random_rate

    def __call__(self, img: np.ndarray):
        assert isinstance(img, np.ndarray), "img must be a numpy array"
        if np.random.rand() < self.random_rate:
            img = cv2.flip(img, 1)
        return img

