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
    'pca': {
        'n_components': 256,
    },
    'pred_col': 'l1',
    'text_col': 'text',
    'split': {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1
    },
    'tfidf_params': {
        'lowercase': 'true',
        'stop_words': 'english',
        'max_features': 10000,
        'min_df': 0,
        'max_df': 0.2,
        'ngram_range': (1, 1)
    },
    'wordvec_params': {
        'lowercase': 'true',
        'stop_words': 'english',
        'max_features': 512,
        'min_df': 5,
        'max_df': 0.2,
        'ngram_range': (1, 1)
    }
}
