# Once we are done with the development, just load the config.yaml here
# If in code, dataclasses would be better, but we have no time for that

CONFIG = {
    'debug': False,
    'debug_info': {
        'head_size': 1024 * 40
    },
    'universal_sentence_encoder': {
        'batch_size': 1024 * 4,
        'url': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    },
    'tfidf': {

    },
    'storage': {
        'initial_matrices': 'data/initial_matrices.pkl'
    },
    'pred_col': 'fraudulent', # name of column with label
    'text_col': 'text', # name of column with text
    "use_f1": True # change to False if you use Accuracy
}

"""
if parameter has to be integer, then it has to be listed in "int" section, otherwise it is float
"""
BAYES_OPT_CONFIG = {
    "steps": 2, # change to 10?
    "init_points": 2,
    "od": {
        "pca": {
            "int": {
                "n_components": [ 64, 512 ],
                "n_iter": [ 1, 10 ]
            }
        },
        "scaled": {},
        "none": {}
    },
    "clf" : {
        "svc": {
            "C": [ 0.5, 10 ]
        },
        "forest": {
            "int": {
                "n_estimators": [ 5, 1000 ],
                "min_samples_split": [ 2, 10 ],
                "min_samples_leaf": [ 1, 10 ]
            },
            "min_weight_fraction_leaf": [ 0, 0.5 ],
            "max_features": [ 0, 1 ],
            "max_samples": [ 0, 1 ]
        },
        "mlp": {
            "int": {
                "hidden": [ 32, 1024 ]
            },
            "alpha": [ 0.00001, 0.001 ],
            "learning_rate_init": [ 0.0001, 0.01 ],
            "beta_1": [ 0.8, 1 ],
            "beta_2": [ 0.8, 1 ]
        }
    }
}