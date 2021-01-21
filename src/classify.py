from collections import defaultdict
from logging import info, debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import CONFIG
from src.utils import OdMatrix, Dataset, train_size


# TODO: Meta classifier for all scikit?
def classify_svc(od_matrices: OdMatrix, original_dfs: Dataset):
    info(f'Classifying svc on {train_size(od_matrices)}')
    res = defaultdict(dict)
    for od, imat in od_matrices.items():  # TODO: Parallelism `with multiprocessing.Pool() as pool`
        for mtype, dfs in imat.items():
            debug(f'{od} {mtype} {train_size(imat)}')
            clf = SVC()
            clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])
            score = clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])
            res[od][mtype] = score
    return res


def classify_forest(od_matrices: OdMatrix, original_dfs: Dataset):
    info(f'Classifying forest on {train_size(od_matrices)}')
    res = defaultdict(dict)
    for od, imat in od_matrices.items():
        for mtype, dfs in imat.items():
            debug(f'{od} {mtype} {train_size(imat)}')
            clf = RandomForestClassifier(n_jobs=-1)
            clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])
            score = clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])
            res[od][mtype] = score
    return res


# TODO: Change to tf.Keras for extra spice
def classify_mlp(od_matrices: OdMatrix, original_dfs: Dataset):
    info(f'Classifying MLP on {train_size(od_matrices)}')
    res = defaultdict(dict)
    for od, imat in od_matrices.items():
        for mtype, dfs in imat.items():
            debug(f'{od} {mtype} {train_size(imat)}')
            clf = MLPClassifier(warm_start=True, hidden_layer_sizes=(128, ), early_stopping=True, learning_rate='adaptive')
            clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])
            score = clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])
            res[od][mtype] = score
    return res


def classify(od_matrices: OdMatrix, original_dfs: Dataset):
    return {
        'svc': classify_svc(od_matrices, original_dfs),
        'forest': classify_forest(od_matrices, original_dfs),
        'mlp': classify_mlp(od_matrices, original_dfs),
    }
