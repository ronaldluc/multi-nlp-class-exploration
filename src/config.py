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
    "use_f1": True  # change to False if you use Accuracy
}

"""
float parameters have to be in float format -> xx.xx
"""
BAYES_OPT_CONFIG = {
    "steps": 13,  # change to 10?
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
        "svc": {
            "C": [0.5, 10.0]
        },
        "forest": {
            "n_estimators": [5, 1000],
            "min_samples_split": [2, 10],
            # "min_samples_leaf": [1, ],
            # "min_weight_fraction_leaf": [ 0.0, 0.5 ],
            # "max_features": [ 0.0, 1.0 ],
            # "max_samples": [ 0.0, 1.0 ]
        },
        "mlp": {
            "hidden": [32, 1024],
            "alpha": [0.00001, 0.001],
            "learning_rate_init": [0.0001, 0.01],
            "beta_1": [0.8, 1 - 1e-4],
            "beta_2": [0.8, 1 - 1e-4]
        }
    }
}
