from logging import debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config import CONFIG
from src.utils import Dataset, train_size


def classify_method(dfs: Dataset, original_dfs: Dataset, method_class, is_test: bool, **kwargs):
    debug(f'classify_method: Classifying {method_class.__name__} on {train_size(dfs):e}')

    clf = method_class(**kwargs)
    clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])

    test_type = "test" if is_test else "val"

    if CONFIG["use_f1"]:
        return f1_score(original_dfs[test_type][CONFIG['pred_col']], clf.predict(dfs[test_type]), average='weighted')
    return clf.score(dfs[test_type], original_dfs[test_type][CONFIG['pred_col']])


# TODO: Meta classifier for all scikit?
def classify_svc_rbf(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='rbf', **kwargs)


def classify_svc_linear(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='linear', **kwargs)


def classify_svc_poly(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='poly', **kwargs)


def classify_svc_sigmoid(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='sigmoid', **kwargs)


def classify_forest(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, RandomForestClassifier, is_test, n_jobs=-1, **kwargs)


# TODO: Change to tf.Keras for extra spice
def classify_mlp(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    hidden_layer_sizes = (kwargs["hidden_1"], kwargs["hidden_2"], kwargs["hidden_3"])
    for i in range(3):
        del kwargs["hidden_{}".format(i+1)]
    return classify_method(od_matrices, original_dfs, MLPClassifier, is_test, warm_start=True,
                           hidden_layer_sizes=hidden_layer_sizes, early_stopping=True, **kwargs)


def classify(od_matrices: Dataset, original_dfs: Dataset, method: str, is_test: bool, **kwargs):
    return {
        'svc_rbf': classify_svc_rbf,
        'svc_linear': classify_svc_linear,
        'svc_poly': classify_svc_poly,
        'svc_sigmoid': classify_svc_sigmoid,
        'forest': classify_forest,
        'mlp': classify_mlp,
    }[method](od_matrices, original_dfs, is_test, **kwargs)
