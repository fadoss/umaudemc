#
# Gather all backends and define functions to access them
#

from .backend.pymc import PyModelChecking
from .backend.nusmv import NuSMV
from .backend.ltsmin import LTSmin
from .backend.bmcalc import BuiltinBackend
from .common import *

supported_logics = {
	'maude':	{'propLogic', 'LTL'},
	'ltsmin':	{'propLogic', 'LTL', 'CTL', 'CTL*', 'Mucalc'},
	'pymc':		{'propLogic', 'LTL', 'CTL', 'CTL*'},
	'nusmv':	{'propLogic', 'LTL', 'CTL'},
	'builtin':	{'propLogic', 'CTL', 'Mucalc'}
}


class MaudeBackend:
	"""The Maude LTL model checker and its strategy-aware extension"""

	def find(self):
		return True

	def check(self, module=None, graph=None, formula_str=None, term=None,
		  strategy=None, opaque=[], full_matchrew=False, get_graph=False,
		  **kwargs):
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
		stats = {'states': graph.getNrStates()}

		if get_graph:
			stats['graph'] = graph

		if not result.holds:
			stats['counterexample'] = (result.leadIn, result.cycle)

		return result.holds, stats


def get_backend(backend):
	"""Get the backend class for a given backend name"""
	if backend == 'maude':
		return MaudeBackend()
	elif backend == 'ltsmin':
		return LTSmin()
	elif backend == 'pymc':
		return PyModelChecking()
	elif backend == 'nusmv':
		return NuSMV()
	elif backend == 'builtin':
		return BuiltinBackend()


def get_backends(backend_arg):
	"""Get the lists of available and unavailable backends give a comma-separated list with their names"""
	available, unavailable = [], []

	for name in backend_arg.split(','):
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
