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
        'lowercase': 'true',
        'stop_words': 'english',
        'max_features': 10000,
        'min_df': 0,
        'max_df': 0.2,
        'ngram_range': (1, 1)
    },
    'storage': {
        'initial_matrices_folder': 'data/tmp',
        'results': 'data/tmp/results.pkl',
        'output': 'data/results_1000.csv',
    },
    'wordvec': {
        'url': 'fasttext-wiki-news-subwords-300',
        'limit_size': 200_000,
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
    "init_points": 3,
    "od": {
        "pca": {
            "n_components": [64, 1000],
            "n_iter": [5, 10]
        },
        "scaled": {},
        "none": {}
    },
    "clf": {
        "svc_rbf": {  # floats are pow(10, X)
            "C": [-1.0, 6.0],
            # "gamma": [1e-10, 1e-2]
        },
        "svc_linear": {  # floats are pow(10, X)
            "C": [-1.0, 6.0]
        },
        "svc_poly": {  # floats are pow(10, X)
            "C": [-1.0, 6.0],
            "degree": [1, 4],
            # "gamma": [1e-10, 1e-2],
            "coef0": [-10, 10]
        },
        "svc_sigmoid": {  # floats are pow(10, X)
            "C": [-2.0, 6.0],
            # "gamma": [1e-10, 1e-2],
            "coef0": [-10, 10]
        },
        "forest": {
            "n_estimators": [20, 1000],
            # "max_depth": [2, 500],    # Most of those hyper params work poorly
            # "min_samples_split": [2, 30],
            # "min_samples_leaf": [1, 10],
            # "min_weight_fraction_leaf": [0.0, 0.25],
            # "max_features": [0.0, 1.0],
            # "min_impurity_decrease": [0.0, 0.1],
            # "max_samples": [0.0, 1.0],
            # "cpp_alpha": [0.0, 0.2]
        },
        "mlp": {  # floats are pow(10, X)
            "hidden_1": [1.1, 2.5],
            "hidden_2": [0.0, 2.0],
            "hidden_3": [0.0, 1.5],
            # "alpha": [-10.0, -2.0],
            "batch_size": [8, 128],
            "learning_rate_init": [-8.0, -1.0],
            # "epsilon": [-10.0, -6.0],
            # "beta_1": [-0.3, 0.0-1e6],
            # "beta_2": [-0.3, 0.0-1e6],
            "n_iter_no_change": [2, 20]
            # "max_iter": [5, 500],
        }
    }
}
