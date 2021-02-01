import pickle
from pathlib import Path
from logging import basicConfig, INFO, info
from pathlib import Path
from pprint import pprint

from src.config import CONFIG, BAYES_OPT_CONFIG
from src.matrices import load_data
from src.pipeline import Pipeline

import pandas as pd

from src.utils import settings_dict2df


def print_config_warnings():
    print("##### CHECK YOUR CONFIG #####")
    print(f"Text column: {CONFIG['text_col']}")
    print(f"Label column: {CONFIG['pred_col']}")
    print(f"Eval metric: {'F1-score' if CONFIG['use_f1'] else 'Accuracy'}")
    print(f"Bayesian optimization steps: {BAYES_OPT_CONFIG['steps']}")
    print("#############################")


if __name__ == "__main__":
    print_config_warnings()
    # data_split('data/data.csv')
    basicConfig(level=INFO)
    data_folder = Path('data')
    dfs_ = load_data(**{name: next(data_folder.glob(f'DB*{name}.csv')) for name in ['train', 'val', 'test']})
    #     # works as well, change it to your liking TODO: put into some config not to overwrite it for each other
    # dfs_ = load_data(train='data/train.csv', val='data/val.csv', test='data/test.csv')

    pipeline = Pipeline(dfs_)
    pipeline.add_prep(["tfidf", 'uce', 'wordvec', 'multi'])
    # pipeline.add_prep(['multi'])

    pipeline.add_od(["none", 'pca', 'scaled'])

    pipeline.add_clf(['svc_rbf', 'svc_poly', 'forest', 'mlp'])
    # pipeline.add_clf(['mlp'])

    # pipeline.create_matrices()
    pipeline.run()

    best_settings = pickle.load(open(Path(CONFIG['storage']['results']), 'rb'))
    pprint(best_settings)
    settings_dict2df(best_settings).to_csv(CONFIG['storage']['output'], index=False)
    info(f"Output successfully generated to {CONFIG['storage']['output']}")
    info('Done')
