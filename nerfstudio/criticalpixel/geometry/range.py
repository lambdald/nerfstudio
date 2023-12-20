import numpy as np
import torch


class UniformRange:

    def __init__(self, low, high) -> None:
        self.low, self.high = low, high

    def sample(self, num_samples):
        steps = np.linspace(0, 1, num_samples) * (self.high - self.low)
        samples = self.low + steps
        return samples, steps

    def get_bound(self):
        return np.array([[self.low, self.high]])


class UniformRangeWithOverlap:

    def __init__(self, low, high, overlap_ratio) -> None:
        self.low = low
        self.high = high
        self.overlap_ratio = overlap_ratio

    def sample(self, num_samples):
        length = self.high - self.low
        cell_size = length / (num_samples * (1 - self.overlap_ratio) + self.overlap_ratio)
        centers = [self.low + cell_size * 0.5]
        for i in range(1, num_samples):
            centers.append(centers[-1] + (1 - self.overlap_ratio) * cell_size)
        return np.array(centers), cell_size

    def get_bound(self):
        return np.array([[self.low, self.high]])