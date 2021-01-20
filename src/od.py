from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.utils import InitMatrix, OdMatrix


# TODO: Maybe merge all scikit transformators and use scikit pipeline?
# TODO: Check https://stackoverflow.com/a/48807797/6253183 to streamline OD and CLF steps
def apply_pca(initial_matrices: InitMatrix) -> InitMatrix:
    res = {}
    # pca = PCA(n_components=CONFIG['pca']['n_components'])   # TODO: Non spare `scipy.sparse.issparse(my_matrix)`
    transformator = TruncatedSVD(n_components=CONFIG['pca']['n_components'])  # for sparse only
    for mtype, dfs in initial_matrices.items():
        transformator.fit(dfs['train'])
        res[mtype] = {dataset: transformator.transform(df) for dataset, df in dfs.items()}
    return res


def apply_scaler(initial_matrices: InitMatrix) -> InitMatrix:
    res = {}
    transformator = StandardScaler(with_mean=False)
    # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    # TODO: perform differently for sparse
    for mtype, dfs in initial_matrices.items():
        transformator.fit(dfs['train'])
        res[mtype] = {dataset: transformator.transform(df) for dataset, df in dfs.items()}
    return res


def apply_od(initial_matrices: InitMatrix) -> OdMatrix:
    return {
        'none': initial_matrices,
        'pca': apply_pca(initial_matrices),
        'scaled': apply_scaler(initial_matrices),   # TODO: Probably change scaled as the "none" default
                                                    # and come up with some other OD
    }
