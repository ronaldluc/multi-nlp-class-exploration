from bayes_opt import BayesianOptimization
from logging import info

from src.classify import classify
from src.matrices import load_data, create_matrices
from src.config import CONFIG, BAYES_OPT_CONFIG
from src.od import apply_od

current_od = None
current_clf = None
dataset = None # dfs after preprocessing
orig_dataset = None # original_dfs

def optimized_function( **kwargs ):
	global current_od, current_clf, dataset
	# print( "kwargs =", kwargs )

	od_kwargs, clf_kwargs = Pipeline.distribute_config( **kwargs )
	# print( f'od_kwargs={od_kwargs}, clf_kwargs={clf_kwargs}')

	od_dfs = apply_od( dataset, current_od, **od_kwargs )
	return classify( od_dfs, orig_dataset, current_clf, **clf_kwargs )

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
		global dataset
		results = {}
		for prep in self.prep_methods:
			dataset = create_matrices( self.dfs, prep )
			for od in self.od_methods:
				for clf in self.clf_methods:
					try:
						best_settings = self._run_step( prep, od, clf )
						results[f"{prep}-{od}-{clf}"] = best_settings
					except Exception as e:
						info( f'Step {prep}-{od}-{clf}: bayes opt failed, {e}' )
						results[f"{prep}-{od}-{clf}"] = "failed"

		print( results )

	def _run_step( self, prep, od, clf ):
		global current_clf, current_od, orig_dataset

		info( f'Running {prep}-{od}-{clf}' )
		orig_dataset = self.dfs
		current_od = od
		current_clf = clf

		od_config = BAYES_OPT_CONFIG["od"].get( current_od )
		clf_config = BAYES_OPT_CONFIG["clf"].get( current_clf )
		# print( od_config, clf_config )

		kwargs = {}
		for k, v in od_config.items():
			if k == "int":
				continue
			kwargs[k] = v
		for k, v in clf_config.items():
			if k == "int":
				continue
			kwargs[k] = v
		if "int" in od_config:
			for k, v in od_config.get( "int" ).items():
				kwargs[k] = v
		if "int" in clf_config:
			for k, v in clf_config.get( "int" ).items():
				kwargs[k] = v

		optimizer = BayesianOptimization( f=optimized_function, pbounds=kwargs )
		optimizer.maximize( init_points=BAYES_OPT_CONFIG["init_points"], n_iter=BAYES_OPT_CONFIG["steps"] )

		info( f"Best setting {optimizer.max}" )
		info( f'Finished {prep}-{od}-{clf}' )

		return optimizer.max

	@staticmethod
	def distribute_config( **kwargs ):
		od_config = BAYES_OPT_CONFIG["od"].get( current_od )
		clf_config = BAYES_OPT_CONFIG["clf"].get( current_clf )
		od_kwargs = {}
		clf_kwargs = {}
		for k, v in kwargs.items():
			if k in od_config:
				od_kwargs[k] = v
			elif k in clf_config:
				clf_kwargs[k] = v
			elif "int" in od_config and k in od_config.get( "int" ):
				od_kwargs[k] = int( round( v ) )
			elif "int" in clf_config and k in clf_config.get( "int" ):
				clf_kwargs[k] = int( round( v ) )
			else:
				info( f'Key {k} is not in any configuration' )

		return od_kwargs, clf_kwargs