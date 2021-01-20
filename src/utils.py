import math
from typing import Dict, Optional, Any, Union
import numpy as np
import pandas as pd

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
