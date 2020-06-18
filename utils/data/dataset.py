import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=True):
        self.sessions = sessions
        index = create_index(sessions)  # columns: sessionId, labelIndex
        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        return seq, label

    def __len__(self):
        return len(self.index)

