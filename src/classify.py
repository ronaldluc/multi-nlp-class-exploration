from collections import defaultdict
from logging import info, debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from src.config import CONFIG
from src.utils import Dataset, train_size


def classify_method(dfs: Dataset, original_dfs: Dataset, method_class, **kwargs):
    info(f'classify_method: Classifying {method_class.__name__} on {train_size(dfs)}')
    # info(f'kwargs: {kwargs}')

    res = defaultdict(dict)
    clf = method_class(**kwargs)
    clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])

    if CONFIG["use_f1"]:
        return f1_score( original_dfs['test'][CONFIG['pred_col']], clf.predict( dfs['test'] ) )
    return clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])

# TODO: Meta classifier for all scikit?
def classify_svc(od_matrices: Dataset, original_dfs: Dataset, **kwargs):
    return classify_method( od_matrices, original_dfs, SVC, **kwargs )

def classify_forest(od_matrices: Dataset, original_dfs: Dataset, **kwargs):
    return classify_method( od_matrices, original_dfs, RandomForestClassifier, n_jobs=-1, **kwargs )

# TODO: Change to tf.Keras for extra spice
def classify_mlp(od_matrices: Dataset, original_dfs: Dataset, **kwargs):
    hidden_layer_sizes = (kwargs["hidden"],)
    del kwargs["hidden"]
    return classify_method( od_matrices, original_dfs, MLPClassifier, warm_start=True, hidden_layer_sizes=hidden_layer_sizes, early_stopping=True, learning_rate='adaptive', **kwargs )

def classify(od_matrices: Dataset, original_dfs: Dataset, method: str, **kwargs):
    return {
        'svc': classify_svc,
        'forest': classify_forest,
        'mlp': classify_mlp,
    }[method](od_matrices, original_dfs, **kwargs)
