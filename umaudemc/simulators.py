#
# Statistical simulators
#

import random

from .common import usermsgs, maude


def collect_vars(term, varset):
	"""Find a variable in the term"""

	for arg in term.arguments():
		if arg.isVariable():
			varset.add(arg)
		else:
			collect_vars(arg, varset)


def parse_hole_term(module, term_str):
	"""Parse a term with a single variable"""

	term = module.parseTerm(term_str)

	if term is None:
		return None, None

	# Collect all variables in the term
	varset = set()
	collect_vars(term, varset)

	if len(varset) > 1:
		usermsgs.print_warning('The observation "{message}" '
		                       'contains more than one variable.')

	elif not varset:
		return term, None

	# We do not check whether the variable is in the appropriate kind
	return term, varset.pop()


class BaseSimulator:
	"""Base class for all simulators"""

	def __init__(self, initial):
		self.initial = initial
		self.module = initial.symbol().getModule()
		self.state = self.initial

		self.step = 0

		self.bool_sort = self.module.findSort('Bool')
		self.true = self.module.parseTerm('true', self.bool_sort.kind())

		self.obs_cache = {}  # Cache of parsed observations

	def restart(self):
		"""Restart simulator"""

		self.state = self.initial
		self.step = 0

	def get_time(self):
		"""Get simulation time (number of steps)"""

		return float(self.step)

	def rval(self, observation):
		"""Evaluate an observation on the current state of the simulation"""

		if observation == 'steps':
			return float(self.step)

		if observation == 'time':
			return self.get_time()

		t, var = self.obs_cache.get(observation, (None, None))
		if not t:
			t, var = parse_hole_term(self.module, observation)
			self.obs_cache[observation] = (t, var)
		subs = maude.Substitution({var: self.state})
		t = subs.instantiate(t)
		t.reduce()

		return float(t == self.true) if t.getSort() == self.bool_sort else float(t)


class StrategyStepSimulator(BaseSimulator):
	"""Simulator where a strategy is the step"""

	def __init__(self, initial, strategy):
		super().__init__(initial)

		self.strategy = strategy

	def next_step(self):
		"""Perform a step of the simulation"""

		next_state, _ = next(self.state.srewrite(self.strategy), (None, None))

		if next_state is not None:
			self.state = next_state
			self.step += 1


def all_children(graph, state):
	"""All children of a state in a graph"""

	children, next_state, index = [], graph.getNextState(state, 0), 0

	while next_state != -1:
		children.append(next_state)
		index += 1
		next_state = graph.getNextState(state, index)

	return children


class UmaudemcSimulator(BaseSimulator):
	"""Simulator that uses the probability assigners of pcheck"""

	def __init__(self, initial, graph, assigner):
		super().__init__(initial)

		self.state_nr = 0
		self.graph = graph
		self.assigner = assigner

	def restart(self):
		"""Restart the simulator"""

		super().restart()
		self.state_nr = 0

	def next_step(self):
		"""Perform a step of the simulation"""

		successors = all_children(self.graph, self.state_nr)

		if successors:
			probs = self.assigner(self.graph, self.state_nr, successors)
			self.state_nr, = random.choices(successors, probs)
			self.state = self.graph.getStateTerm(self.state_nr)
			self.step += 1

	@staticmethod
	def new(initial, strategy=None, assigner='uniform', opaque=()):
		"""Construct a UmaudemcSimulator, but may fail if the assigner is not valid"""

		from . import probabilistic as pb

		if strategy:
			graph = pb.maude.StrategyRewriteGraph(initial, strategy, opaque)
		else:
			graph = pb.maude.RewriteGraph(initial)

		graph.strategyControlled = strategy is not None
		distr, found = pb.get_local_assigner(initial.symbol().getModule(), assigner)

		if not found:
			usermsgs.print_error(f'Unknown probability assignment method {assigner}.')

		return UmaudemcSimulator(initial, graph, distr) if distr else None


class StrategyPathSimulator(BaseSimulator):
	"""Simulator based on the strategy where steps are rewrites"""

	def __init__(self, module, initial, strategy=None):
		super().__init__(initial)

		from .pyslang import StratCompiler, RandomRunner

		ml = maude.getModule('META-LEVEL')
		sc = StratCompiler(module, ml, use_notify=True, ignore_one=True)
		p = sc.compile(ml.upStrategy(strategy))

		self.runner = RandomRunner(p, initial)

	def restart(self):
		"""Restart simulator"""

		super().restart()
		self.runner.reset(self.initial)

	def next_step(self):
		"""Perform a step of the simulation"""

		next_state = self.runner.run()

		if next_state:
			self.state = next_state
			self.step += 1


class StrategyDTMCSimulator(BaseSimulator):
	"""Simulator based on the strategy where steps are rewrites"""

	def __init__(self, module, initial, strategy=None):
		super().__init__(initial)

		from .pyslang import StratCompiler, MarkovRunner, BadProbStrategy

		ml = maude.getModule('META-LEVEL')
		sc = StratCompiler(module, ml, use_notify=True, ignore_one=True)
		p = sc.compile(ml.upStrategy(strategy))

		try:
			self.graph = MarkovRunner(p, initial).run()
			self.node = self.graph

		except BadProbStrategy as bps:
			usermsgs.print_error(bps)

	def restart(self):
		"""Restart simulator"""

		super().restart()
		self.node = self.graph

	def next_step(self):
		"""Perform a step of the simulation"""

		nd_opts, qt_opts, next_node = len(self.node.children), len(self.node.child_choices), None

		if nd_opts + qt_opts > 1:
			usermsgs.print_warning('Unquantified nondeterministic choices are present in the strategy.')

		if nd_opts:
			next_node, *_ = self.node.children

		elif qt_opts:
			choice, *_ = self.node.child_choices
			next_node = random.choices(list(choice.keys()), choice.values())[0]

		if next_node:
			self.node = next_node
			self.state = self.node.term
			self.step += 1


class PMaudeSimulator(BaseSimulator):
	"""Python-based PMaude simulator"""

	def __init__(self, initial, getTime, tick, nat_kind, val):
		super().__init__(initial)

		self.getTimeOp = getTime
		self.tick = tick
		self.nat_kind = nat_kind
		self.val = val

		# Prepares the initial term
		self.state = self.initial.copy()
		self.state.rewrite()

		# Try to find Maude's random symbol for calculating random(1). If the
		# PMaude specification only reduces random(0), even after resetting
		# the random seed for a new simulation, it will take the same value
		self.random = self.module.findSymbol('random', (nat_kind,), nat_kind)

		if self.random:
			self.random = self.random(self.module.parseTerm('1'))

	def restart(self):
		"""Restart simulator"""

		# PMaude uses Maude's random symbol, which is memoryless
		# and deterministic for a fixed seed, so we need a new seed
		if self.random:
			self.random.copy().reduce()
		maude.setRandomSeed(random.getrandbits(31))

		self.state = self.initial.copy()
		self.state.rewrite()
		self.step = 0

	def get_time(self):
		"""Get simulation time (internal PMaude time)"""

		time = self.getTimeOp(self.state)
		time.reduce()

		return float(time)

	def rval(self, observation):
		"""Evaluate an observation on the current state"""

		# Observations in PMaude are usually indexed by integers
		if isinstance(observation, int) or isinstance(observation, float):
			obs = self.val(self.module.parseTerm(str(int(observation))), self.state)
			obs.reduce()

			return float(obs)

		return super().rval(observation)

	def next_step(self):
		"""Perform a step of the simulation"""

		self.state = self.tick(self.state)
		self.state.rewrite()
		self.step += 1

	@staticmethod
	def new(module, initial, strategy):

		if strategy:
			usermsgs.print_warning('the strategy will be ignored for the PMaude method.')

		# PMaude specifications are based in a set of functions that we collect here
		config_kind = module.findSort('Config')

		if not config_kind:
			usermsgs.print_error('not a valid PMaude specification, '
			                     'the "Config" sort is not defined.')
			return None

		config_kind = config_kind.kind()
		float_kind = module.findSort('Float').kind()

		getTime = module.findSymbol('getTime', (config_kind, ), float_kind)

		if not getTime:
			usermsgs.print_error('not a valid PMaude specification, '
			                     'the "getTime" function is not defined.')
			return None

		tick = module.findSymbol('tick', (config_kind, ), config_kind)

		if not tick:
			usermsgs.print_error('not a valid PMaude specification, '
			       	             'the "tick" function is not defined.')
			return None

		nat_kind = module.findSort('Nat').kind()
		val = module.findSymbol('val', (nat_kind, config_kind), float_kind)

		if not val:
			usermsgs.print_error('not a valid PMaude specification, '
			                     'the "val" function is not defined.')
			return None

		return PMaudeSimulator(initial, getTime, tick, nat_kind, val)


def get_simulator(method, data):
	"""Get the simulator for the given assignment method"""

	# The default method depends on whether a strategy is given
	if method is None:
		method = 'step' if data.strategy else 'uniform'

	if method == 'step':
		return StrategyStepSimulator(data.term, data.strategy)

	if method == 'strategy':
		return StrategyPathSimulator(data.module, data.term, data.strategy)

	if method == 'strategy-full':
		return StrategyDTMCSimulator(data.module, data.term, data.strategy)

	if method == 'pmaude':
		return PMaudeSimulator.new(data.module, data.term, data.strategy)

	return UmaudemcSimulator.new(data.term, data.strategy, method, opaque=data.opaque)
