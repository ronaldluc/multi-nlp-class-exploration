import pickle
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from time import time
from typing import List, Dict

import pandas as pd
import gensim.parsing.preprocessing as pp
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pyarrow  # to check for read/to_feather
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from stemming.lovins import stem
import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
from logging import log, ERROR, INFO, WARN, basicConfig, DEBUG, info

from src.config import CONFIG
from src.utils import InitMatrix, Dataset, OdMatrix


def iterated_lovins_stemmer(word):
    if len(word) <= 3:
        return word
    try:
        new_word = stem(word)
    except IndexError:
        new_word = word
    while (len(new_word) > 2) and (len(new_word) != len(word)):
        word = new_word
        try:
            new_word = stem(word)
        except IndexError:
            new_word = word
    return new_word


def stem_string(string):
    stemmed_tokens = [iterated_lovins_stemmer(token.lower()) for token in string.split(" ")]
    return " ".join(stemmed_tokens)


def create_matrices(dfs: Dict[str, pd.DataFrame]) -> InitMatrix:
    """
    dfs must contain 'train' key
    all df must contain column 'text'
    The reason why not to store it all in the one DF
    https://stackoverflow.com/questions/56618078/how-to-insert-a-multidimensional-numpy-array-to-pandas-column
    """
    info(f'Computing initial matrices on {dfs["train"].shape}')
    return {
        'tfidf': create_tfidf(dfs),
        'uce': create_use(dfs),
        'wordvec': create_wordvec(dfs),
    }


def create_tfidf(dfs: Dataset) -> Dataset:
    """Compute TF-IDF based doc embedding"""
    info(f'Computing tf-idf on {dfs["train"].shape}')
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase='true',
                                 preprocessor=stem_string, stop_words='english',
                                 max_features=10000, dtype=np.float32, min_df=10,
                                 max_df=0.2, ngram_range=(1, 1))
    vectorizer.fit(dfs['train'].text)
    return {name: vectorizer.transform(df.text) for name, df in dfs.items()}


def create_use(dfs: Dataset) -> Dataset:
    """Compute universal sentence encodings based doc embedding"""
    info(f'Computing universal sentence encodings on {dfs["train"].shape}')
    model = hub.load(CONFIG['universal_sentence_encoder']['url'])
    info(f"Module {CONFIG['universal_sentence_encoder']['url']} loaded")

    return {name: np.concatenate([model(batch_df)
                                  for batch_number, batch_df in
                                  df.text.groupby(
                                      np.arange(len(df)) // CONFIG['universal_sentence_encoder']['batch_size'])])
            for name, df in dfs.items()}


def create_wordvec(dfs: Dataset) -> Dataset:
    """Compute word-vector based doc embedding"""
    info(f'Computing word vectors on {dfs["train"].shape}')
    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase='true',  # put into config
                                 preprocessor=stem_string, stop_words='english',
                                 max_features=512, dtype=np.float32, min_df=10,
                                 max_df=0.2, ngram_range=(1, 3))
    vectorizer.fit(dfs['train'].text)
    return {name: vectorizer.transform(df.text) for name, df in dfs.items()}


def load_df(path) -> pd.DataFrame:
    if path.suffix == '.csv' and not path.with_suffix('.feather').exists():
        df = pd.read_csv(path)
        df.to_feather(path.with_suffix('.feather'))
    else:
        df = pd.read_feather(path.with_suffix('.feather'))
    if CONFIG['debug']:
        return df.head(CONFIG['debug_info']['head_size'])
    return df


def load_data(**paths: Path) -> Dataset:
    """train: Path, val: Path, test: Path"""
    return {name: load_df(Path(path)) for name, path in paths.items()}


def apply_pca(initial_matrices: InitMatrix) -> InitMatrix:
    res = {}
    # pca = PCA(n_components=CONFIG['pca']['n_components'])   # TODO: Non spare `scipy.sparse.issparse(my_matrix)`
    transformator = TruncatedSVD(n_components=CONFIG['pca']['n_components'])  # for sparse only
    for mtype, dfs in initial_matrices.items():
        transformator.fit(dfs['train'])
        res[mtype] = {dataset: transformator.transform(df) for dataset, df in dfs.items()}
    return res


# TODO: Maybe merge all scikit transformators and use scikit pipeline?
# TODO: Check https://stackoverflow.com/a/48807797/6253183 to streamline OD and CLF steps
def apply_scaler(initial_matrices: InitMatrix) -> InitMatrix:
    res = {}
    transformator = StandardScaler(with_mean=False)
    # ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
    # TODO: perform differently for sparse
    for mtype, dfs in initial_matrices.items():
        transformator.fit(dfs['train'])
        res[mtype] = {dataset: transformator.transform(df) for dataset, df in dfs.items()}
    return res


def apply_od(initial_matrices: InitMatrix) -> OdMatrix:
    return {
        'none': initial_matrices,
        'pca': apply_pca(initial_matrices),
        'scaled': apply_scaler(initial_matrices),
    }


def classify_svc(od_matrices: OdMatrix, original_dfs: Dataset):
    res = defaultdict(dict)
    for od, imat in od_matrices.items():
        for mtype, dfs in initial_matrices.items():
            clf = SVC()
            clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])
            score = clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])
            res[od][mtype] = score
    return res


# TODO: Meta classifier for all scikit?
def classify_forest(od_matrices: OdMatrix, original_dfs: Dataset):
    res = defaultdict(dict)
    for od, imat in od_matrices.items():
        for mtype, dfs in initial_matrices.items():
            clf = RandomForestClassifier()
            clf.fit(dfs['train'], original_dfs['train'][CONFIG['pred_col']])
            score = clf.score(dfs['test'], original_dfs['test'][CONFIG['pred_col']])
            res[od][mtype] = score
    return res


def classify(od_matrices: OdMatrix, original_dfs: Dataset):
    return {
        'svc': classify_svc(od_matrices, original_dfs),
        'forest': classify_forest(od_matrices, original_dfs),
    }


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    data_folder = Path('../data')
    dfs_ = load_data(**{name: next(data_folder.glob(f'*{name}.csv')) for name in ['train', 'val', 'test']})
    # works as well
    # dfs = load_data(train='dataset.csv', val='deltaset.csv', test='gammaset.csv')
    initial_matrices = create_matrices(dfs_)
    pickle.dump(initial_matrices, open(CONFIG['storage']['initial_matrices'], 'wb'))
    initial_matrices = pickle.load(open(CONFIG['storage']['initial_matrices'], 'rb'))
    od_matrices = apply_od(initial_matrices)
    results = classify(od_matrices, dfs_)
    pprint(results)
    info('Done')
