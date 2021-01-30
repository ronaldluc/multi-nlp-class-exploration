import math
from datetime import timedelta
from logging import info
from time import time
from typing import Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from os.path import isfile

Dataset = Dict[str, Union[pd.DataFrame, np.ndarray]]
"""[train/val/test] -> np.ndarray"""

InitMatrix = Dict[str, Dataset]
"""[matrix type][train/val/test] -> np.ndarray"""

OdMatrix = Dict[str, InitMatrix]
"""[OD method][matrix type][train/val/test] -> np.ndarray"""


def total_size(x: Dict):
    if isinstance(x, dict):
        return sum([total_size(x_) for x_ in x.values()])
    return math.prod(x.shape)


def train_size(x: Dict):
    if isinstance(x, dict):
        if 'train' in x:
            return math.prod(x['train'].shape)
        return sum([total_size(x_) for x_ in x.values()])


def data_split(filepath: str):
    df = pd.read_csv(filepath)
    size = df.iloc[:, 0].size
    # df.l1 = pd.factorize(df.l1)[0]

    train, validate, test = np.split(df.sample(frac=1), [int(.6 * size), int(.8 * size)])

    train.to_csv('data/train.csv', index=False)
    validate.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)


def nested_max(obj):
    """Returns maximum value in a nested dictionary"""
    if type(obj) == float or type(obj) == int or type(obj) == np.float64:
        return obj
    return max([nested_max(val) for key, val in obj.items()])


class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.log_len = len(str(total))
        self.start = time()
        self.last = time()
        self.exp_average = 0
        self.factor = 3

    def log(self, done=None, left=None):
        done = done if done is not None else (self.total - left)
        left = (self.total - done)
        delta = time() - self.start
        self.exp_average = self.exp_average + ((time() - self.last) - self.exp_average) / min(done + 1e-4, self.factor)
        self.last = time()
        eta = (delta / (done + 1e-4)) * left
        eta_delta = abs(eta - self.exp_average * left)
        info(f'{done:{self.log_len}}/{self.total:{self.log_len}} done\t'
             f'rate: {done / (delta + 1e-4):5.2}\t'
             f'ETA: {str(timedelta(seconds=eta-eta_delta)).split(".")[0]}-'
             f'{str(timedelta(seconds=eta+eta_delta)).split(".")[0]}')
