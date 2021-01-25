import math
import pickle
from pathlib import Path
from pprint import pprint

from logging import basicConfig, DEBUG, info

from src.classify import classify
from src.matrices import load_data, create_matrices
from src.config import CONFIG
from src.od import apply_od
from src.utils import InitMatrix, total_size, data_split

if __name__ == "__main__":
    # data_split('data/data.csv')
    basicConfig(level=DEBUG)
    data_folder = Path('data')
    dfs_ = load_data(**{name: next(data_folder.glob(f'DB*{name}.csv')) for name in ['train', 'val', 'test']})
    # works as well, change it to your liking TODO: put into some config not to overwrite it for each other
    # dfs_ = load_data(train='data/train.csv', val='data/val.csv', test='data/test.csv')
    initial_matrices = create_matrices(dfs_)

    pickle.dump(initial_matrices, open(CONFIG['storage']['initial_matrices'], 'wb'))
    initial_matrices_: InitMatrix = pickle.load(open(CONFIG['storage']['initial_matrices'], 'rb'))
    size = total_size(initial_matrices_)
    info(f'Loaded initial matrices {size:10} {int(size ** (1 / 2)):10}^2')
    od_matrices = apply_od(initial_matrices_)
    results = classify(od_matrices, dfs_)
    pprint(results)
    info('Done')
