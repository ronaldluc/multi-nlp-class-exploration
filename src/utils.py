from typing import Dict, Optional, Any, Union
import numpy as np
import pandas as pd

Dataset = Dict[str, Union[pd.DataFrame, np.ndarray]]
"""[train/val/test] -> np.ndarray"""

InitMatrix = Dict[str, Dataset]
"""[matrix type][train/val/test] -> np.ndarray"""

OdMatrix = Dict[str, InitMatrix]
"""[OD method][matrix type][train/val/test] -> np.ndarray"""
