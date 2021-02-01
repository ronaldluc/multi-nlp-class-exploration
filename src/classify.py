from logging import debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow import keras as K

from src.config import CONFIG
from src.utils import Dataset, train_size, exp10_floats, sparse2numpy


def classify_method(od_matrices: Dataset, original_dfs: Dataset, method_class, is_test: bool, **kwargs):
    debug(f'classify_method: Classifying {method_class.__name__} on {train_size(od_matrices):e}')

    clf = method_class(**kwargs)
    clf.fit(od_matrices['train'], original_dfs['train'][CONFIG['pred_col']])

    test_type = "test" if is_test else "val"

    if CONFIG["use_f1"]:
        return f1_score(original_dfs[test_type][CONFIG['pred_col']],
                        clf.predict(od_matrices[test_type]), average='weighted')
    return clf.score(od_matrices[test_type], original_dfs[test_type][CONFIG['pred_col']])


# TODO: Meta classifier for all scikit?
def classify_svc_rbf(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='rbf', **exp10_floats(kwargs))


def classify_svc_linear(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='linear', **exp10_floats(kwargs))


def classify_svc_poly(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='poly', **exp10_floats(kwargs))


def classify_svc_sigmoid(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, SVC, is_test, kernel='sigmoid', **exp10_floats(kwargs))


def classify_forest(od_matrices: Dataset, original_dfs: Dataset, is_test: bool, **kwargs):
    return classify_method(od_matrices, original_dfs, RandomForestClassifier, is_test, n_jobs=-1, **kwargs)


def classify_mlp(od_matrices: Dataset, original_dfs: Dataset, is_test: bool,
                 hidden_1, hidden_2, hidden_3, **kwargs):
    hidden_layer_sizes = [int(10 ** x) for x in [hidden_1, hidden_2, hidden_3] if x > 1.0]
    return classify_method(od_matrices, original_dfs, MLPClassifier, is_test, warm_start=True,
                           hidden_layer_sizes=hidden_layer_sizes, early_stopping=True, **exp10_floats(kwargs))


def classify_mlp_keras(od_matrices: Dataset, original_dfs: Dataset, is_test: bool,
                       hidden_1, hidden_2, hidden_3, patience=2, dropout=0.1, **kwargs):
    label_encoder = LabelEncoder()
    label_encoder.fit(original_dfs['train'][CONFIG['pred_col']])
    od_matrices = sparse2numpy(od_matrices)

    hidden_layer_sizes = [int(10 ** x) for x in [hidden_1, hidden_2, hidden_3] if x > 1.0]
    input_shape = od_matrices['train'].shape[-1]
    output_size = len(original_dfs['train'][CONFIG['pred_col']].unique())

    model = K.models.Sequential(
        [K.layers.Input(shape=(input_shape,)), ])
    for size in hidden_layer_sizes:
        model.add(K.layers.Dense(size, 'relu'))
        model.add(K.layers.Dropout(dropout))
    model.add(K.layers.Dense(output_size, 'softmax'))

    callbacks = [K.callbacks.EarlyStopping(patience=patience, min_delta=1e-4),
                 K.callbacks.ReduceLROnPlateau(patience=patience // 2, factor=0.3)]
    loss = K.losses.sparse_categorical_crossentropy
    optimizer = K.optimizers.Adam(learning_rate=0.0033)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(od_matrices['train'], label_encoder.transform(original_dfs['train'][CONFIG['pred_col']]),
              epochs=256, verbose=0, callbacks=callbacks,
              validation_data=(od_matrices['val'], label_encoder.transform(original_dfs['val'][CONFIG['pred_col']])),
              **kwargs)
    test_type = "test" if is_test else "val"

    if CONFIG["use_f1"]:
        return f1_score(label_encoder.transform(original_dfs[test_type][CONFIG['pred_col']]),
                        model.predict(od_matrices[test_type]), average='weighted')
    return model.evaluate(od_matrices[test_type], label_encoder.transform(original_dfs[test_type][CONFIG['pred_col']]),
                          verbose=0)[1]


def classify(od_matrices: Dataset, original_dfs: Dataset, method: str, is_test: bool, **kwargs):
    return {
        'svc_rbf': classify_svc_rbf,
        'svc_linear': classify_svc_linear,
        'svc_poly': classify_svc_poly,
        'svc_sigmoid': classify_svc_sigmoid,
        'forest': classify_forest,
        'mlp': classify_mlp_keras,
    }[method](od_matrices, original_dfs, is_test, **kwargs)
