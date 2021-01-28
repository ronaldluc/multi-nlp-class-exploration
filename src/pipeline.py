from bayes_opt import BayesianOptimization
from logging import info

from src.classify import classify
from src.matrices import load_data, create_matrices
from src.config import CONFIG, BAYES_OPT_CONFIG
from src.od import apply_od

class Pipeline:

	def __init__( self, dfs ):
		self.dfs = dfs
		self.prep_methods = []
		self.od_methods = []
		self.clf_methods = []

	def add_prep( self, name ):
		self.prep_methods.append( name )
	def add_od( self, name ):
		self.od_methods.append( name )
	def add_clf( self, name ):
		self.clf_methods.append( name )

	def run( self ):
		total_settings = len( self.prep_methods ) * len( self.od_methods ) * len( self.clf_methods )
		index = 1
		results = {}
		for prep in self.prep_methods:
			dataset = create_matrices( self.dfs, prep )
			for od in self.od_methods:
				for clf in self.clf_methods:
					pipe_name = f"{prep}-{od}-{clf}"
					try:
						info( f'Running {pipe_name} ({index}/{total_settings})' )
						best_settings = self._run_step( dataset, prep, od, clf )
						results[pipe_name] = best_settings
						info( f"Best setting {best_settings}" )
						info( f'Finished {prep}-{od}-{clf}' )
					except Exception as e:
						info( f'Step {prep}-{od}-{clf}: bayes opt failed, {e}' )
						results[pipe_name] = "failed"

					if results[pipe_name] != "failed":
						results[pipe_name]["test_score"] = self._test_settings( self.dfs, dataset, od, clf, results[pipe_name]["params"] )

					index += 1

		return results

	def _run_step( self, prep_dataset, prep, od, clf ):
		od_config = BAYES_OPT_CONFIG["od"].get( od )
		clf_config = BAYES_OPT_CONFIG["clf"].get( clf )

		kwargs = {}
		for k, v in od_config.items():
			if type( v[0] ) == int:
				v = [ int( round( x ) ) for x in v ]
			kwargs[k] = v
		for k, v in clf_config.items():
			if type( v[0] ) == int:
				v = [ int( round( x ) ) for x in v ]
			kwargs[k] = v

		optimizer = BayesianOptimization( f=self.create_optimized_function( self.dfs, prep_dataset, od, clf ), pbounds=kwargs )
		optimizer.maximize( init_points=BAYES_OPT_CONFIG["init_points"], n_iter=BAYES_OPT_CONFIG["steps"] )
		return optimizer.max

	def _test_settings( self, orig_dataset, prep_dataset, od, clf, settings ):
		od_kwargs, clf_kwargs = Pipeline.distribute_config( od, clf, **settings )
		od_dfs = apply_od( prep_dataset, od, **od_kwargs )
		return classify( od_dfs, orig_dataset, clf, is_test=True, **clf_kwargs )

	def create_optimized_function( self, orig_dataset, prep_dataset, od, clf ):
		def optimized_function( **kwargs ):
			od_kwargs, clf_kwargs = Pipeline.distribute_config( od, clf, **kwargs )
			print( f'od_kwargs={od_kwargs}, clf_kwargs={clf_kwargs}')

			od_dfs = apply_od( prep_dataset, od, **od_kwargs )
			return classify( od_dfs, orig_dataset, clf, is_test=False, **clf_kwargs )

		return optimized_function

	@staticmethod
	def distribute_config( od, clf, **kwargs ):
		od_config = BAYES_OPT_CONFIG["od"].get( od )
		clf_config = BAYES_OPT_CONFIG["clf"].get( clf )
		od_kwargs = {}
		clf_kwargs = {}
		for k, v in kwargs.items():
			if k in od_config:
				if type( od_config.get( k )[0] ) == int:
					v = int( round( v ) )
				od_kwargs[k] = v
			elif k in clf_config:
				if type( clf_config.get( k )[0] ) == int:
					v = int( round( v ) )
				clf_kwargs[k] = v
			else:
				info( f'Key {k} is not in any configuration' )

		return od_kwargs, clf_kwargs