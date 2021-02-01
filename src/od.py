import copy
from logging import debug

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.utils import Dataset
from src.utils import train_size


# TODO: Maybe merge all scikit transformers and use scikit pipeline?
# TODO: Check https://stackoverflow.com/a/48807797/6253183 to streamline OD and CLF steps
def apply_pca(dfs: Dataset, **kwargs) -> Dataset:
    debug(f'Applying PCA on {train_size(dfs):e}')
    # pca = PCA(n_components=CONFIG['pca']['n_components'])   # TODO: Non spare `scipy.sparse.issparse(my_matrix)`
    kwargs['n_components'] = min(kwargs['n_components'], dfs['train'].shape[-1] - 1)
    transformer = TruncatedSVD(**kwargs)  # for sparse only
    transformer.fit(dfs['train'])
    return {dataset: transformer.transform(df) for dataset, df in dfs.items()}


def apply_scaler(dfs: Dataset, **kwargs) -> Dataset:
    debug(f'Applying Scaler on {train_size(dfs):e}')
    transformer = StandardScaler(with_mean=False, **kwargs)
    # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    # TODO: perform differently for sparse
    transformer.fit(dfs['train'])
    return {dataset: transformer.transform(df) for dataset, df in dfs.items()}


def apply_none(dfs: Dataset, **kwargs) -> Dataset:
    return copy.deepcopy(dfs)


def apply_od(dfs: Dataset, method: str, **kwargs) -> Dataset:
    return {
        'none': apply_none,
        'pca': apply_pca,
        'scaled': apply_scaler,  # TODO: Probably change scaled as the "none" default
        # and come up with some other OD
    }[method](dfs, **kwargs)
