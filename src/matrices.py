from logging import info
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow_hub as hub
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
                                 max_df=0.2, ngram_range=(1, 1))
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
