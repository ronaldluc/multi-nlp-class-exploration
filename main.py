from pathlib import Path
from logging import basicConfig, INFO
from pathlib import Path
from pprint import pprint

from src.config import CONFIG, BAYES_OPT_CONFIG
from src.matrices import load_data
from src.pipeline import Pipeline


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
    pipeline.add_prep("tfidf").add_prep("uce").add_prep("wordvec")

    pipeline.add_od("none").add_od("pca").add_od("scaled")

    pipeline.add_clf("svc").add_clf("forest").add_clf("mlp")

    # pipeline.create_matrices()
    best_settings = pipeline.run()
    pprint(best_settings)

    # initial_matrices = create_matrices(dfs_)
    # pickle.dump(initial_matrices, open(CONFIG['storage']['initial_matrices'], 'wb'))
    # initial_matrices_: InitMatrix = pickle.load(open(CONFIG['storage']['initial_matrices'], 'rb'))
    # size = total_size(initial_matrices_)
    # info(f'Loaded initial matrices {size:10} {int(size ** (1 / 2)):10}^2')
    # od_matrices = apply_od(initial_matrices_)
    # results = classify(od_matrices, dfs_)
    # pprint(results)
    # info('Done')
