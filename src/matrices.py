from logging import info
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import re
import tensorflow_hub as hub
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.lovins import stem

from gensim.models.fasttext import load_facebook_vectors
from gensim.test.utils import datapath
import gensim.downloader as api

from src.config import CONFIG
from src.utils import InitMatrix, Dataset


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
    vectorizer = TfidfVectorizer(**CONFIG['tfidf_params'], strip_accents='unicode', preprocessor=stem_string,
                                 dtype=np.float32)
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
    # cap_path = datapath(CONFIG['wordvec_params']['link'])
    # model = load_facebook_vectors(cap_path)
    model = api.load(CONFIG['wordvec_params']['link'])

    return {name: get_wordvec(df, model) for name, df in dfs.items()}


def get_wordvec(df, model):
    """Get word-vector based doc embedding given text and model"""
    # One of these should work
    return df.text.apply(lambda doc: model[simple_preprocess(doc)].mean(axis=0))
    return np.stack(df.text.apply(lambda doc: model[simple_preprocess(doc)].mean(axis=0)))
    return df.text.apply(lambda doc: model[simple_preprocess(doc)].mean(axis=0)).to_numpy()
    return np.stack([model[simple_preprocess(doc)].mean(axis=0) for doc in df.text])


def load_df(path) -> pd.DataFrame:
    if path.suffix == '.csv' and not path.with_suffix('.feather').exists():
        df = pd.read_csv(path)
        df.to_feather(path.with_suffix('.feather'))
    else:
        df = pd.read_feather(path.with_suffix('.feather'))
    if CONFIG['debug']:
        return df.head(CONFIG['debug_info']['head_size'])
    return df.rename(columns={CONFIG['text_col']: 'text'})


def load_data(**paths: Path) -> Dataset:
    """train: Path, val: Path, test: Path"""
    return {name: load_df(Path(path)) for name, path in paths.items()}
