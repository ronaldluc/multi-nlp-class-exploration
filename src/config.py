# Once we are done with the development, just load the config.yaml here
# If in code, dataclasses would be better, but we have no time for that

CONFIG = {
    'debug': True,
    'debug_info': {
        'head_size': 1024
    },
    'universal_sentence_encoder': {
        'batch_size': 1024,
        'url': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    },
    'tfidf': {

    },
    'storage': {
        'initial_matrices': 'data/tmp'
    },
    'pred_col': 'l1',  # name of column with label
    'text_col': 'text',  # name of column with text
    "use_f1": False  # change to False if you use Accuracy
}

"""
float parameters have to be in float format -> xx.xx
"""
BAYES_OPT_CONFIG = {
    "steps": 2,  # change to 10?
    "init_points": 4,
    "od": {
        "pca": {
            "n_components": [64, 512],
            "n_iter": [1, 10]
        },
        "scaled": {},
        "none": {}
    },
    "clf": {
        "svc_rbf": {
            "C": [1e-1, 1e10],
            "gamma": [1e-10, 1e-2]
        },
        "svc_linear": {
            "C": [1e-1, 1e10]
        },
        "svc_poly": {
            "C": [1e-1, 1e10],
            "degree": [2, 10],
            "gamma": [1e-10, 1e-2],
            "coef0": [-10.0, 10.0]
        },
        "svc_sigmoid": {
            "C": [1e-2, 1e10],
            "gamma": [1e-10, 1e-2],
            "coef0": [-10.0, 10.0]
        },
        "forest": {
            "n_estimators": [5, 1000],
            "max_depth": [2, 500],
            "min_samples_split": [0.0, 0.25],
            "min_samples_leaf": [0.0, 0.25],
            "min_weight_fraction_leaf": [0.0, 0.25],
            "max_features": [0.0, 1.0],
            "min_impurity_decrease": [0.0, 0.1],
            "max_samples": [0.0, 1.0],
            "cpp_alpha": [0.0, 0.2]
        },
        "mlp": {
            "hidden_1": [32, 2048],
            "hidden_2": [16, 1024],
            "hidden_3": [8, 512],
            "alpha": [1e-10, 1e-2],
            "batch_size": [8, 1024],
            "learning_rate_init": [1e-10, 1e-1],
            "max_iter": [5, 500],
            "beta_1": [0.5, 1-1e-6],
            "beta_2": [0.5, 1-1e-6],
            "epsilon": [1e-10, 1e-6],
            "n_iter_no_change": [2, 50]
        }
    }
}
