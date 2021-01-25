import math
from typing import Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from src.config import CONFIG
from logging import info, debug

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
    info(f'Splitting dataset into train: {CONFIG["split"]["train"]}, val: {CONFIG["split"]["val"]}, '
         f'test: {CONFIG["split"]["test"]}')
    in_df = pd.read_csv(filepath)
    size = len(in_df)

    for name, df in zip(['train', 'val', 'test'], np.split(in_df.sample(frac=1), [int(.6 * size), int(.8 * size)])):
        df.to_csv(f'data/{name}.csv', index=False)
