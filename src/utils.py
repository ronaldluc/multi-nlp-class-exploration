import math
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
    size = df.iloc[:,0].size
    #df.l1 = pd.factorize(df.l1)[0]

    train, validate, test = np.split(df.sample(frac=1), [int(.6*size), int(.8*size)])

    train.to_csv('data/train.csv', index=False)
    validate.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
