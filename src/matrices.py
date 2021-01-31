from logging import info
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim import downloader
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.lovins import stem

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


def create_matrices(dfs: Dict[str, pd.DataFrame], method: str) -> InitMatrix:
    """
    dfs must contain 'train' key
    all df must contain column 'text'
    The reason why not to store it all in the one DF
    https://stackoverflow.com/questions/56618078/how-to-insert-a-multidimensional-numpy-array-to-pandas-column
    """
    info(f'Computing initial matrices on {dfs["train"].shape}')
    return {
        'tfidf': create_tfidf,
        'uce': create_uce,
        'wordvec': create_wordvec,
    }[method](dfs)


def create_tfidf(dfs: Dataset) -> Dataset:
    """Compute TF-IDF based doc embedding"""
    info(f'Computing tf-idf on {dfs["train"].shape}')
    vectorizer = TfidfVectorizer(strip_accents='unicode', preprocessor=stem_string, dtype=np.float32, **CONFIG['tfidf'])
    vectorizer.fit(dfs['train'].text)
    return {name: vectorizer.transform(df.text) for name, df in dfs.items()}


def create_uce(dfs: Dataset) -> Dataset:
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
    path = downloader.load(CONFIG['wordvec']['url'], return_path=True)
    w2v = KeyedVectors.load_word2vec_format(path, binary=False, limit=CONFIG['wordvec']['limit_size'])
    return {name: np.stack(df.text.apply(
        lambda doc: w2v[[w for w in simple_preprocess(doc) if w in w2v]].mean(axis=0)
    )) for name, df in dfs.items()}


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
