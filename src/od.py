from logging import debug

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.utils import InitMatrix, OdMatrix, train_size
from src.utils import Dataset

import copy


# TODO: Maybe merge all scikit transformers and use scikit pipeline?
# TODO: Check https://stackoverflow.com/a/48807797/6253183 to streamline OD and CLF steps
def apply_pca(initial_matrices: Dataset, **kwargs) -> Dataset:
    debug(f'Applying PCA on {train_size(initial_matrices)}')
    # pca = PCA(n_components=CONFIG['pca']['n_components'])   # TODO: Non spare `scipy.sparse.issparse(my_matrix)`
    transformer = TruncatedSVD(**kwargs)  # for sparse only
    transformer.fit(initial_matrices['train'])
    return {dataset: transformer.transform(df) for dataset, df in initial_matrices.items()}


def apply_scaler(dfs: Dataset, **kwargs) -> Dataset:
    debug(f'Applying Scaler on {train_size(dfs)}')
    transformer = StandardScaler(with_mean=False, **kwargs)
    # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    # TODO: perform differently for sparse
    transformer.fit(dfs['train'])
    return {dataset: transformer.transform(df) for dataset, df in dfs.items()}


def apply_none(initial_matrices: Dataset, **kwargs) -> Dataset:
    return copy.deepcopy(initial_matrices)


def apply_od(initial_matrices: Dataset, method: str, **kwargs) -> Dataset:
    debug(f'Applying OD on {train_size(initial_matrices)}')
    return {
        'none': apply_none,
        'pca': apply_pca,
        'scaled': apply_scaler,  # TODO: Probably change scaled as the "none" default
        # and come up with some other OD
    }[method](initial_matrices, **kwargs)
