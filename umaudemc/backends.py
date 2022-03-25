#
# Gather all backends and define functions to access them
#

from .common import *

# Logics supported by each backend
supported_logics = {
	'maude':   {'propLogic', 'LTL'},
	'ltsmin':  {'propLogic', 'LTL', 'CTL', 'CTL*', 'Mucalc'},
	'pymc':    {'propLogic', 'LTL', 'CTL', 'CTL*'},
	'nusmv':   {'propLogic', 'LTL', 'CTL'},
	'spot':    {'propLogic', 'LTL'},
	'spin':    {'propLogic', 'LTL'},
	'builtin': {'propLogic', 'CTL', 'Mucalc'}
}

# List of backends in the default order
DEFAULT_BACKENDS = 'maude,ltsmin,pymc,nusmv,spot,spin,builtin'

# Backends that support the generation of counterexamples
counterexample_backends = {'maude', 'nusmv', 'spot', 'spin'}

# Backend that directly support the Kleene star semantics of the iteration
kleene_backends = {'spot'}


STATS_FORMAT = {
	'states':   '{} system state',
	'rewrites': '{} rewrite',
	'game':     '{} game state',
	'buchi':    '{} Büchi state'
}


class MaudeBackend:
	"""The Maude LTL model checker and its strategy-aware extension"""

	def find(self):
		return True

	def check(self, module=None, graph=None, formula_str=None, term=None,
	          strategy=None, opaque=(), full_matchrew=False, get_graph=False, **_):
		"""Check an LTL formula using the Maude LTL model checker"""

		formula = module.parseTerm(formula_str)

		# This should not happen because it was already parsed
		if formula is None:
			usermsgs.print_error('The LTL formula cannot be parsed.')
			return None, None

		# Create the graph if it was not given
		if graph is None:
			graph = maude.StrategyRewriteGraph(term, strategy, opaque, full_matchrew) \
				if strategy else maude.RewriteGraph(term)

		result = graph.modelCheck(formula)

		if result is None:
			return None, None

		# Build the statistics dictionary
		stats = {
			'states': graph.getNrStates(),
			'rewrites': graph.getNrRewrites(),
			'buchi': result.nrBuchiStates
		}

		if strategy:
			stats['real_states'] = graph.getNrRealStates()

		if get_graph:
			stats['graph'] = graph

		if not result.holds:
			stats['counterexample'] = (list(result.leadIn), list(result.cycle))

		return result.holds, stats


def get_backend(backend):
	"""Get the backend class for a given backend name"""
	if backend == 'maude':
		return MaudeBackend()
	elif backend == 'ltsmin':
		from .backend.ltsmin import LTSmin
		return LTSmin()
	elif backend == 'pymc':
		from .backend.pymc import PyModelChecking
		return PyModelChecking()
	elif backend == 'nusmv':
		from .backend.nusmv import NuSMV
		return NuSMV()
	elif backend == 'spot':
		from .backend.spot import SpotBackend
		return SpotBackend()
	elif backend == 'spin':
		from .backend.spin import Spin
		return Spin()
	elif backend == 'builtin':
		from .backend.bmcalc import BuiltinBackend
		return BuiltinBackend()


def get_backends(backend_arg):
	"""Get the lists of available and unavailable backends give a comma-separated list with their names"""
	available, unavailable = [], []

	# If the argument is None, take it from the environment or by default
	if backend_arg is None:
		backend_arg = os.getenv('UMAUDEMC_BACKEND')

		if backend_arg is None:
			backend_arg = DEFAULT_BACKENDS

	backend_names = backend_arg.split(',') if isinstance(backend_arg, str) else backend_arg

	for name in backend_names:
		handler = get_backend(name)

		if handler:
			(available if handler.find() else unavailable).append((name, handler))
		else:
			usermsgs.print_warning(f'Unsupported backend "{name}". Ignoring it.')

	return available, unavailable


def backend_for(backends, logic):
	"""First supported backend for the given logic"""

	valid = ((name, handle) for name, handle in backends
	         if logic in supported_logics[name])
	return next(valid, (None, None))


def advance_counterexample(backends):
	"""Advance backends that provide counterexamples"""

	return ([b for b in backends if b[0] in counterexample_backends] +
	        [b for b in backends if b[0] not in counterexample_backends])


def advance_kleene(backends):
	"""Advance backends that directly support the Kleene star iteration"""

	return ([b for b in backends if b[0] in kleene_backends] +
	        [b for b in backends if b[0] not in kleene_backends])


def format_statistics(stats):
	"""Format the statistics provided by the backends"""

	# Format the integral statistic messages
	params = [msg.format(stats[key]) + ('s' if stats[key] > 1 else '')
	          for key, msg in STATS_FORMAT.items() if key in stats]

	sset = stats.get('sset')
	states = stats.get('states')

	if sset is not None:
		params.append(f'holds in {len(sset)}/{states} states')

	return ', '.join(params)
