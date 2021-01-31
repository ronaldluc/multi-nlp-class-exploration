import pickle
from itertools import product
from logging import info, error
from pathlib import Path
from random import gauss, shuffle

from bayes_opt import BayesianOptimization

from src.classify import classify
from src.config import CONFIG, BAYES_OPT_CONFIG
from src.matrices import create_matrices
from src.od import apply_od
from src.utils import ProgressLog


class Pipeline:

    def __init__(self, dfs):
        self.dfs = dfs
        self.prep_methods = []
        self.od_methods = []
        self.clf_methods = []

    def add_prep(self, name):
        self.prep_methods.append(name)
        return self

    def add_od(self, name):
        self.od_methods.append(name)
        return self

    def add_clf(self, name):
        self.clf_methods.append(name)
        return self

    def create_matrices(self):
        for prep in self.prep_methods:
            dataset = create_matrices(self.dfs, prep)
            pickle.dump(dataset, open(Path(CONFIG['storage']['initial_matrices_folder']) / f'{prep}.pkl', 'wb'))

    def run(self):
        results = {}
        search_space = list(product(self.prep_methods, self.od_methods, self.clf_methods))
        shuffle(search_space)
        progress = ProgressLog(len(search_space))
        for index, pipe_def in enumerate(search_space):     # TODO: Do want to parallelize?
            prep, od, clf = pipe_def
            info(f'Running {pipe_def}')
            progress.log(done=index)
            try:
                dataset = pickle.load(open(Path(CONFIG['storage']['initial_matrices_folder']) / f'{prep}.pkl', 'rb'))
                best_settings = self._run_step(dataset, prep, od, clf)
                results[pipe_def] = best_settings
                info(f"Best setting {best_settings}")
                info(f'Finished {prep}-{od}-{clf}')
            except Exception as e:
                error(f'Step {prep}-{od}-{clf}: bayes opt failed, {e}')
                results[pipe_def] = "failed"

            if results[pipe_def] != "failed":
                results[pipe_def]["test_score"] = self._test_settings(self.dfs, dataset, od, clf,
                                                                      results[pipe_def]["params"])
            pickle.dump(results, open(Path(CONFIG['storage']['results']), 'wb'))

        return results

    def _run_step(self, prep_dataset, prep, od, clf):
        kwargs = {**BAYES_OPT_CONFIG["od"].get(od), **BAYES_OPT_CONFIG["clf"].get(clf)}
        optimized_kwargs = {k: v for k, v in kwargs.items() if len(v) == 2}
        locked_kwargs = {k: v for k, v in kwargs.items() if len(v) == 1}

        optimizer = BayesianOptimization(f=self.create_optimized_function(self.dfs, prep_dataset, od, clf,
                                                                          **locked_kwargs),
                                         pbounds=optimized_kwargs)
        optimizer.maximize(init_points=BAYES_OPT_CONFIG["init_points"], n_iter=BAYES_OPT_CONFIG["steps"])
        result = optimizer.max
        result['params'].update(locked_kwargs)
        return result

    @staticmethod
    def _test_settings(orig_dataset, prep_dataset, od, clf, settings):
        od_kwargs, clf_kwargs = Pipeline.distribute_config(od, clf, **settings)
        od_dfs = apply_od(prep_dataset, od, **od_kwargs)
        return classify(od_dfs, orig_dataset, clf, is_test=True, **clf_kwargs)

    @staticmethod
    def create_optimized_function(orig_dataset, prep_dataset, od, clf, **locked_kwargs):
        def optimized_function(**optimized_kwargs):
            od_kwargs, clf_kwargs = Pipeline.distribute_config(od, clf, **locked_kwargs, **optimized_kwargs)

            od_dfs = apply_od(prep_dataset, od, **od_kwargs)
            return classify(od_dfs, orig_dataset, clf, is_test=False, **clf_kwargs) + gauss(0, 1e-6)

        return optimized_function

    @staticmethod
    def distribute_config(od, clf, **kwargs):
        od_config = BAYES_OPT_CONFIG["od"].get(od)
        clf_config = BAYES_OPT_CONFIG["clf"].get(clf)
        od_kwargs = {k: round(v) if isinstance(od_config[k][0], int) else v
                     for k, v in kwargs.items() if k in od_config}
        clf_kwargs = {k: round(v) if isinstance(clf_config[k][0], int) else v
                      for k, v in kwargs.items() if k in clf_config}

        return od_kwargs, clf_kwargs
