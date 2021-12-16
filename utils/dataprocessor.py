from collections import Counter, OrderedDict
from utils.negativesampler import NegativeSampler
import itertools
import pickle
import numpy as np


class DataProcessor:
    def __init__(self, file_path: str, n_words: int = 1000):
        self.tokenizer = Tokenizer(file_path)

        self.sampling_probabilities = self.tokenizer.sampling_probabilities()[:n_words]

        self.negative_probabilities = self.tokenizer.negative_probabilities()[:n_words]

        self.negativeSampler = NegativeSampler(self.negative_probabilities)

    def sampling(self, indices: List[int]):

        keep = []

        for idx in indices:
            r_num = np.random.uniform(low=0.0, high=1.0)
            if idx < self.n_words and self.sampling_probabilities[idx] > r_num:
                keep.append(True)
            else:
                keep.append(False)

        return keep

    def process_sentence(self, sentence: list, window_size: int = 2):
        samples = []
        labels = []

        sequence = self.tokenizer.sentence_to_sequence(sentence)

        keep = self.sampling(sequence)


class Tokenizer:
    def __init__(self, file_path: str):

        with open(file_path, "rb") as file:
            self.corpus = [text[0] for text in pickle.load(file)]

        self.word_counts = OrderedDict(
            Counter(list(itertools.chain.from_iterable(self.corpus))).most_common()
        )

        self.n_tokens = sum(list(self.word_counts.values()))

        self.word_index = {w: idx + 1 for (idx, w) in enumerate(list(self.word_counts))}
        self.vocab_size = len(self.word_index.keys())

        self.frequencies = [
            self.word_counts[word] / self.n_tokens for word in self.word_index
        ]

    def sampling_probabilities(self):
        probabilities = [
            (np.sqrt(freq / 0.001) + 1) * (0.001 / freq) for freq in self.frequencies
        ]

        return probabilities

    def negative_probabilities(self):
        negative_probabilities = [freq ** 0.75 for freq in self.frequencies]
        sum_prob = sum(negative_probabilities)

        negative_probabilities = [
            neg_prob / sum_prob for neg_prob in negative_probabilities
        ]

        return negative_probabilities
