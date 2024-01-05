#
# umaudemc API to be used by external programs or extensions
#

import os
import sys

import maude

from . import backends as _backends
from . import common as _common
from . import counterprint as _counterprint
from . import formatter as _formatter
from . import formulae as _formulae
from . import grapher as _grapher
from . import probabilistic as _probabilistic
from . import quatex as _quatex
from . import simulators as _simulators
from . import statistical as _statistical
from . import usermsgs as _usermsgs
from . import wrappers as _wrappers
from .backend import prism as _prism
from .backend import storm as _storm


DEFAULT_BACKENDS = tuple(_backends.DEFAULT_BACKENDS.split(','))


# The API allows providing terms as objects, and they may need to be passed to
# LTSmin in the form of string. The mixfix representation of the terms may
# cause ambiguities and parsing problems, so the term is converted recursively
# to prefix unambiguous form if maude 0.5 is not available, otherwise
# prettyPrint is used.

if hasattr(maude.Term, 'prettyPrint'):
	def _term2str(term):
		"""Convert any term to a string in prefix form"""
		return term.prettyPrint(maude.PRINT_WITH_PARENS | maude.PRINT_DISAMBIG_CONST)

else:
	PSEUDOSYMBOLS = frozenset({'<Strings>', '<Qids>', '<Floats>'})

	def _term2str(term):
		"""Convert any term to a string in prefix form"""

		symbol = term.symbol()
		symbol_name = str(symbol)
		arguments = list(term.arguments())

		if symbol.arity() == 0:
			return str(term) if symbol_name in PSEUDOSYMBOLS \
			                 else f'({symbol_name}).{term.getSort()}'

		# Avoid some ambiguities with associative operators
		elif symbol.arity() == 2 and len(arguments) > 2:
			args = _term2str(arguments[0])
			for arg in arguments[1:]:
				args = f'{symbol_name}({args}, {arg})'
			return args

		else:
			args = ','.join(f'({_term2str(arg)})' for arg in arguments)
			return f'{symbol_name}({args})'


class MaudeModel:
	"""Model of a Maude (strategy-controlled) rewriting system"""

	def __init__(self, initial, strategy=None, filename=None, module=None,
	             metamodule=None, opaque=(), biased_matchrew=True,
	             already_loaded=False, single_use=False):
		"""
		Generate a Maude model for model-checking.

		:param initial: Initial term
		:type initial: str or maude.Term
		:param strategy: Strategy to control rewriting
		:type strategy: str or maude.StrategyExpression or None
		:param filename: Name of the file to be loaded
		:type filename: str or None
		:param module: Module where to model check
		:type module: str or maude.Module or None
		:param metamodule: Metarepresentation of the module where to model check
		(parsed in module)
		:type metamodule: str or maude.Term or None
		:param opaque: List of opaque strategies
		:type opaque: list of str
		:param biased_matchrew: Whether the biased matchrew feature is enabled
		:param biased_matchrew: bool
		:param already_loaded: Whether the file should not be loaded again
		:param already_loaded: bool
		:param single_use: Whether a single use of the model with the graph method
		is intended. Otherwise, graphs will be cached between calls to check
		:param single_use: bool
		"""

		# File name

		self.filename = filename
		needs_loading = (not isinstance(module, maude.Module)
		                 and not isinstance(initial, maude.Term)
		                 and not already_loaded)

		if needs_loading:
			if self.filename is None:
				raise ValueError('filename must not be empty if not already loaded')
			else:
				maude.load(self.filename)

		# Module

		if module is None:
			if isinstance(initial, maude.Term) and metamodule is None:
				self.module = initial.symbol().getModule()
			else:
				self.module = maude.getCurrentModule()
			self.module_str = str(self.module)
		elif isinstance(module, str):
			self.module = maude.getModule(module)
			self.module_str = module
		elif isinstance(module, maude.Module):
			self.module = module
			self.module_str = str(module)
		else:
			raise TypeError(f"unexpected type '{type(module).__name__}' for module")

		# Metamodule

		if isinstance(metamodule, str):
			self.metamodule = self.module.parseTerm(metamodule)
			self.module = maude.downModule(self.metamodule)
		elif isinstance(metamodule, maude.Term):
			self.metamodule = metamodule
			self.module = maude.downModule(metamodule)
		else:
			self.metamodule = None

		# Initial term

		if isinstance(initial, str):
			self.initial = self.module.parseTerm(initial)
			self.initial_str = initial
		elif isinstance(initial, maude.Term):
			self.initial = initial
			self.initial_str = str(initial)
		else:
			raise TypeError(f"unexpected type '{type(module).__name__}' for term")

		# Strategy expression

		if isinstance(strategy, str):
			self.strategy = self.module.parseStrategy(strategy)
			self.strategy_str = strategy
		elif isinstance(strategy, maude.StrategyExpression):
			self.strategy = strategy
			self.strategy_str = str(strategy)
		else:
			self.strategy = None
			self.strategy_str = None

		# Opaque strategies and biased_matchrew

		self.opaque = opaque
		self.biased_matchrew = biased_matchrew

		# Build the parser

		self.parser = _formulae.Parser()
		self.parser.set_module(self.module, metamodule=self.metamodule)

		# Look for the Formula sort

		formula_sort = self.module.findSort('Formula')
		if formula_sort is None:
			raise ValueError('the given module is not prepared for model checking')
		self.formula_kind = self.module.findSort('Formula').kind()

		# Graphs (wrapped or not)

		if self.strategy is None:
			self.graph = maude.RewriteGraph(self.initial)
		else:
			self.graph = maude.StrategyRewriteGraph(self.initial, self.strategy,
			                                        self.opaque, self.biased_matchrew)

		self.wgraphs = {}
		self.single_use = single_use

	def _make_data(self):
		"""Make an InitialData structure from the model content"""

		data = _common.InitialData()

		data.module = self.module
		data.strategy = self.strategy
		data.term = self.initial
		data.opaque = self.opaque
		data.full_matchrew = not self.biased_matchrew

		return data

	def check(self, formula, purge_fails='default', merge_states='default',
	          backends=DEFAULT_BACKENDS, formula_str=None, logic=None,
	          depth=-1, timeout=None, usermsgs=_usermsgs, extra_args=()):
		"""
		Model check a given temporal formula.

		:param formula: Formula
		:type formula: str or list or maude.Term
		:param purge_fails: Whether failed states must be purged (by default,
			this value is selected depending on the logic)
		:type purge_fails: bool or str
		:param merge_states: Whether successors with a common term should be merged
			(edge, value or none, by default selected depending on the logic)
		:type merge_states: str
		:param backends: Prioritized list of model checking backends to be used
		:type backends: str or list of str
		:param formula_str: Formula given a string (in case formula is a list)
		:type formula_str: str or None
		:param logic: Logic (in case formula is a list)
		:type logic: str
		:param depth: Depth bound on the rewriting graph (only works for some backends)
		:type depth: int
		:param timeout: Timeout for the model-checking task (only works for some backends)
		:type timeout: int or None
		:param usermsgs: Partially override default user message printing functions
		:type usermsgs: An object with print_error, print_warning and print_info methods
		:param extra_args: Additional arguments to be passed to the backend
		:type extra_args: list of str
		:returns: the model-checking result and a dictionary with statistics
		"""

		# Formula

		if isinstance(formula, str):
			pyformula, logic = self.parser.parse(formula, opaques=self.opaque)
			formula_str = formula

		elif isinstance(formula, maude.Term):
			formula_str = formula_str if formula_str else _term2str(formula)
			pyformula, _ = self.parser.parse(formula_str)
			logic = 'LTL'

		elif isinstance(formula, list):
			pyformula = formula
			formula_str = formula_str if formula_str else _formulae.formula_list2str(formula)

			if logic is None:
				raise ValueError('logic cannot be None is the formula is given a list')

		else:
			raise ValueError('formula as been given in an unknown format')

		# Backends (LTSmin cannot be used if no filename is
		# known, and depth excludes LTSmin and Maude)

		if isinstance(backends, str):
			real_backends = backends.split(',')
		else:
			real_backends = list(backends)

		if self.filename is None and 'ltsmin' in real_backends:
			real_backends.remove('ltsmin')

		if depth > 0:
			if 'ltsmin' in real_backends:
				real_backends.remove('ltsmin')
			if 'maude' in real_backends:
				real_backends.remove('maude')

		available, unavailable = _backends.get_backends(real_backends)
		name, backend = _backends.backend_for(available, logic)

		if name is None:
			usermsgs.print_error('No available backend supports the given formula.')
			return None, None

		aprops = set()
		_formulae.collect_aprops(pyformula, aprops)

		# Graph variants (we store them to save work when checking multiple times)

		if self.strategy is not None and (not self.single_use or depth > 0):
			# The argument purge_fails can be a Boolean
			if isinstance(purge_fails, bool):
				purge_fails = 'yes' if purge_fails else 'no'

			# Decide the purge_fails and merge_states values if not explicitly given
			purge, merge = _common.default_model_settings(logic, purge_fails, merge_states,
			                                              self.strategy,
			                                              tableau=name in ['nusmv', 'pymc'])
			wgraph = self.wgraphs.get((purge, merge, depth > 0))

			# The graph is not stored in the cache
			if wgraph is None or (0 < depth != self._get_depth(wgraph)):
				# Wrap the graph to limit its depth
				wgraph = _wrappers.BoundedGraph(self.graph, depth) if depth > 0 else self.graph

				# Create and store the wrapped graph
				wgraph = _wrappers.wrap_graph(wgraph, purge, merge)
				self.wgraphs[(purge, merge, depth > 0)] = wgraph

				# In case both purge and merge are used, we keep the purged
				# graph (which is the graph attribute of the MergedGraph)
				if purge == 'yes' and merge != 'no':
					self.wgraphs[('yes', 'no', depth > 0)] = wgraph.graph
		else:
			wgraph = self.graph

		holds, stats = backend.check(module=self.module,
		                             module_str=self.module_str,
		                             metamodule_str=_term2str(self.metamodule) if self.metamodule else None,
		                             term=self.initial,
		                             term_str=self.initial_str,
		                             strategy=self.strategy,
		                             strategy_str=self.strategy_str,
		                             opaque=self.opaque,
		                             full_matchrew=not self.biased_matchrew,
		                             formula=pyformula,
		                             formula_str=formula_str,
		                             logic=logic,
		                             labels=self.parser.labels,
		                             filename=self.filename,
		                             graph=None if self.single_use and depth <= 0 else wgraph,
		                             aprops=aprops,
		                             get_graph=True,
		                             extra_args=extra_args,
		                             purge_fails=purge_fails,
		                             merge_states=merge_states,
		                             timeout=timeout)

		if holds is not None:
			stats['backend'] = name
			stats['logic'] = logic

		return holds, stats

	def pcheck(self, formula, purge_fails='default', merge_states='default',
	           backends=('prism', 'storm'), timeout=None, usermsgs=_usermsgs,
	           extra_args=(), assign='uniform', steps=False, reward=None):
		"""
		Probabilistic model checking of a given temporal formula.

		:param formula: Formula
		:type formula: str
		:param purge_fails: Whether failed states must be purged (by default,
			this value is selected depending on the logic)
		:type purge_fails: bool or str
		:param merge_states: Whether successors with a common term should be merged
			(edge, value or none, by default selected depending on the logic)
		:type merge_states: str
		:param backends: Prioritized list of model checking backends to be used
		:type backends: str or list of str
		:param timeout: Timeout for the model-checking task
		:type timeout: int or None
		:param usermsgs: Partially override default user message printing functions
		:type usermsgs: An object with print_error, print_warning and print_info methods
		:param extra_args: Additional arguments to be passed to the backend
		:type extra_args: list of str
		:param assign: Probability assignment method
		:type assign: string
		:param steps: Whether the expected number of steps should be calculated
		:type steps: bool
		:param reward: Reward term to calculate its expected value
		:type reward: str or maude.Term or None
		:returns: the probabilistic model-checking result and a dictionary with statistics
		"""

		# Select the model-checking backend
		backend = None

		if isinstance(backends, str):
			backends = backends.split(',')

		for name in backends:
			backend = _storm.StormBackend() if name == 'storm' else _prism.PRISMBackend()

			if backend.find():
				break
			else:
				backend = None

		if backend is None:
			usermsgs.print_error('No available probabilistic backend.')
			return None, None

		# Select the probability assignment method
		data = self._make_data()

		graph = _probabilistic.get_probabilistic_graph(data, assign,
		                                               purge_fails=purge_fails,
		                                               merge_states=merge_states)

		# Parse the temporal formula
		pparser = _formulae.ProbParser()
		pparser.set_module(self.module, self.metamodule)
		pformula, aprops = pparser.parse(formula)

		# Handle the reward
		if reward is not None:
			if isinstance(reward, str):
				reward = self.module.parseTerm(reward)

			if reward is None:
				usermsgs.print_warning('The reward term cannot be parsed. It will be ignored.')
				reward = None

			else:
				reward = _probabilistic.RewardEvaluator.new(reward, self.initial.getSort().kind())

		result, stats = backend.check(module=self.module,
		                              formula=pformula,
		                              logic='CTL',
		                              aprops=aprops,
		                              extra_args=extra_args,
		                              cost=steps,
		                              reward=reward,
		                              graph=graph)

		return result, stats

	def scheck(self, quatex, assign='uniform', alpha=0.05, delta=0.5, block=30, nsims=(30, None),
	           seed=None, jobs=1, usermsgs=_usermsgs, verbose=False):
		"""
		Statistical model checking of a given QuaTEx expression

		:param quatex: Quantitative temporal expression (file name or file object)
		:type quatex: str or text file object
		:param assign: Probability assignment method
		:type assign: string
		:param alpha: Complement of the confidence level (probability outside the confidence interval)
		:type alpha: float (between 0.0 and 1.0)
		:param delta: Maximum admissible radius for the confidence interval
		:type delta: float (positive)
		:param block: Number of simulations before checking the confidence interval
		:type block: int (positive)
		:param nsims: Number of simulations (tuple of lower and upper limits)
		:type nsims: (int or None, int or None)
		:param seed: Random seed
		:type seed: int
		:type jobs: Number of parallel simulation threads
		:type jobs: int
		:param usermsgs: Partially override default user message printing functions
		:type usermsgs: An object with print_error, print_warning and print_info methods
		:param verbose: Enable verbose messages about the simulation state between blocks
		:type verbose: bool
		:returns: the probabilistic model-checking result and a dictionary with statistics
		"""

		# Parse the QuaTEx query
		if isinstance(quatex, str) and os.path.exists(quatex):
			with open(quatex) as quatex_file:
				program = _quatex.parse_quatex(quatex_file, filename=quatex)
		else:
			program = _quatex.parse_quatex(quatex)

		if not program:
			return None

		# Check the simulation parameters
		if not (0 <= alpha <= 1) or delta < 0 or block <= 0:
			usermsgs.print_error(f'Wrong simulation parameters: alpha, delta, or block are outside bounds.')
			return None

		# No queries in the input file
		if not program.nqueries:
			return {'nsims': 0, 'queries': ()}

		# Get the simulator for the given assignment method
		if not (simulator := _simulators.get_simulator(assign, self._make_data())):
			return None

		min_sim, max_sim = nsims

		# Call the statistical model checker
		num_sims, qdata = _statistical.check(program, simulator,
		                                     seed, alpha, delta, block, min_sim, max_sim,
		                                     jobs, verbose)

		# Return cleaned-up version of the result
		return _statistical.qdata_to_dict(num_sims, qdata, program)

	def _get_depth(self, wgraph):
		"""Get the depth of a bounded depth graph"""

		while wgraph.graph != self.graph:
			wgraph = wgraph.graph

		return wgraph.depth

	def format_statistics(self, stats):
		"""
		Format the statistics produced by the model checkers

		:param stats: Statistics produced by the check method
		:type stats: dict
		:return: a formatted text with statistics
		:rtype: str
		"""

		return _backends.format_statistics(stats)

	def _prepare_formatters(self, sformat, eformat):
		"""Prepare the state and edge formatters, which may be given as a string or a function"""

		sformat_str = sformat if isinstance(sformat, str) else None
		eformat_str = eformat if isinstance(eformat, str) else None

		slabel, elabel = _formatter.get_formatters(sformat_str, eformat_str, self.graph.strategyControlled)

		if sformat is not None and sformat_str is None:
			slabel = sformat
		if eformat is not None and eformat_str is None:
			elabel = eformat

		return slabel, elabel

	def print_counterexample(self, stats, printer=None, sformat=None, eformat=None):
		"""
		Pretty print a counterexample for a previous check (nothing is printed
		if no counterexample has been found)

		:param stats: Statistics produced by the check method
		:type stats: dict
		:param printer: Printer for the counterexample
		:type printer: a counterexample printer or None
		:param sformat: State formatter or format string
		:type sformat: str or function or None
		:param eformat: Edge formatter or format string
		:type eformat: str or function or None
		:rtype: str
		"""

		counterexample = stats.get('counterexample')
		graph = stats.get('graph', self.graph)

		if counterexample is None:
			return

		if printer is None:
			printer = _counterprint.SimplePrinter()

		slabel, elabel = self._prepare_formatters(sformat, eformat)

		_counterprint.print_counterexample(graph, counterexample, (printer, slabel, elabel))

	def print_graph(self, outfile=sys.stdout, sformat=None, eformat=None, depth=-1):
		"""
		Print a graph for this model.

		:param outfile: Text stream where to write the graph (by default, the standard output)
		:type outfile: text stream
		:param sformat: State formatter or format string
		:type sformat: str or function or None
		:param eformat: Edge formatter or format string
		:type eformat: str or function or None
		:param depth: Depth bound for the graph generation
		:type depth: int
		"""

		slabel, elabel = self._prepare_formatters(sformat, eformat)

		grapher = _grapher.DOTGrapher(outfile, slabel=slabel, elabel=elabel)
		grapher.graph(self.graph, bound=depth)
