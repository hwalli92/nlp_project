import random as rd
from itertools import accumulate

"""
Adapted from utility code provided in Assignment #6 
"""


class NegativeSampler:
    def __init__(self, probabilities):
        self.cum_probabilities = list(accumulate(probabilities))
        self.values = range(len(probabilities))

    def sample(self, size: int = 5, exclude: set = None):
        if size > len(self.values):
            raise ValueError(f"Can not select {size} unique elements")
        if exclude is None:
            exclude = set()
        samples = set()
        while len(samples) < size:
            samples.update(
                rd.choices(self.values, cum_weights=self.cum_probabilities, k=size)
            )
            samples = samples.difference(exclude)
        return list(samples)[:size]
