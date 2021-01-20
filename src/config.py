# Once we are done with the development, just load the config.yaml here
# If in code, dataclasses would be better, but we have no time for that

CONFIG = {
    'debug': True,
    'debug_info': {
        'head_size': 1024 * 20
    },
    'universal_sentence_encoder': {
        'batch_size': 1024 * 1,
        'url': 'https://tfhub.dev/google/universal-sentence-encoder/4'
    },
    'tfidf': {

    },
    'storage': {
        'initial_matrices': '../data/initial_matrices.pkl'
    },
    'pca': {
        'n_components': 256,
    },
    'pred_col': 'l2',
}
