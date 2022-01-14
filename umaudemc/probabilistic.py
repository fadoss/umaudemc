#
# Probabilistic model-checking support
#
# Many "probability assigners" are defined to distribute probabilities among the
# successors of each state in the model. Their "nondeterminism" property tells
# whether they yield MDPs or DTMCs. In the first case, the result of these
# assigners is a list of lists of (state id, p) tuples where each sublist
# represents a nondeterministic action. In the second case, the result is
# a single list with a probability number for each successor.
#

from collections import Counter
from itertools import chain

from .common import usermsgs, maude
from .wrappers import create_graph


class QuantitativeResult:
	"""Result of quantitative model checking"""

	QR_BOOLEAN = 1
	QR_NUMBER = 2
	QR_RANGE = 3

	def __init__(self, rtype, value, extra):
		self.rtype = rtype
		self.value = value
		self.extra = extra

	@classmethod
	def make_boolean(cls, value, extra=None):
		return cls(cls.QR_BOOLEAN, value, extra)

	@classmethod
	def make_number(cls, value, extra=None):
		return cls(cls.QR_NUMBER, value, extra)

	@classmethod
	def make_range(cls, first, value, extra=None):
		return cls(cls.QR_RANGE, (first.value, value), (first.extra, extra))

	def __str__(self):
		return str(self.value)


def get_local_assigner(module, name):
	"""Instantiate probability assignment methods by their text descriptions"""

	if name == 'metadata':
		return MetadataAssigner(), True

	elif name == 'mdp-metadata':
		return MetadataMDPAssigner(), True

	elif name.startswith('uaction'):
		return WeightedActionAssigner.parse(name), True

	elif name.startswith('term'):
		return MaudeTermFullAssigner.parse(module, name), True

	elif name.startswith('mdp-term'):
		return MaudeTermMDPAssigner.parse(module, name), True

	elif name == 'uniform':
		return UniformAssigner(), True

	elif name == 'mdp-uniform':
		return UniformMDPAssigner(), True

	else:
		return None, False


class UniformAssigner:
	"""Uniform assignment of probabilities among successors"""

	# Nondeterminism is removed completely
	nondeterminism = False

	@staticmethod
	def __call__(graph, state, successors):
		degree = len(successors)
		return [1.0 / degree] * degree


def _get_transition_stmt(graph, origin, dest):
	"""Get the statement producing the given transition"""

	# If the graph is a strategy-controlled-one, the transition
	# can be a rule application or an opaque strategy
	if graph.strategyControlled:
		trans = graph.getTransition(origin, dest)
		ttype = trans.getType()

		if ttype == maude.StrategyRewriteGraph.OPAQUE_STRATEGY:
			return trans.getStrategy()

		elif ttype == maude.StrategyRewriteGraph.RULE_APPLICATION:
			return trans.getRule()

		else:
			return None

	# Otherwise, the transition is surely a rule application
	else:
		return graph.getRule(origin, dest)


def _parse_metadata(graph, origin, dest):
	"""Parse the metadata for the given transition"""

	# Get the statement producing the transition and its metadata attribute
	stmt = _get_transition_stmt(graph, origin, dest)
	mdata = stmt.getMetadata() if stmt else None

	# If the metadata attribute is absent, weight 1 is considered by default
	if mdata is None:
		return 1

	# If the metadata attribute cannot be parsed as a float, 1 is considered
	try: return float(mdata)
	except ValueError:
		return 1


class MetadataAssigner:
	"""Distribute the probability among the successors according
	   to the weights in the metadata attribute"""

	# Nondeterminism is completely avoided
	nondeterminism = False

	@staticmethod
	def __call__(graph, state, successors):
		# Get the weights from the metadata attributes (assuming 1 by default)
		weights = [_parse_metadata(graph, state, next_state) for next_state in successors]
		# Normalize the probabilities
		total = sum(weights)
		return [w / total for w in weights]


def _get_stmt_label(graph, origin, dest):
	"""Get the label of a statement"""

	stmt = _get_transition_stmt(graph, origin, dest)

	if stmt is None:
		return stmt

	elif hasattr(stmt, 'getLabel'):
		return stmt.getLabel()

	else:
		return stmt.getName()


def _assign_weights_by_actions(actions, weights, successors):
	"""Get a dictionary from actions to lists of probabilities"""

	# Grouped weights
	grouped = {action: [] for action in actions}

	for k, w in enumerate(weights):
		grouped[actions[k]].append((w, successors[k]))

	# Normalize weights to probabilities by groups
	for gr in grouped.values():
		total = sum(w for w, _ in gr)
		for k, (w, s) in enumerate(gr):
			gr[k] = (w / total), s

	return grouped.values()


class MetadataMDPAssigner:
	"""Like MetadataAssigner but actions are nondeterministic"""

	# This procedure gives a Markov decision process
	nondeterminism = True

	@staticmethod
	def __call__(graph, state, successors):
		# Actions of the successors
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]
		# Weights of the successors (1 by default)
		weights = [_parse_metadata(graph, state, next_state) for next_state in successors]

		return _assign_weights_by_actions(actions, weights, successors)


class WeightedActionAssigner:
	""""Weighted assignment of probabilisties among actions and then uniform among successors"""

	# Nondeterminism is completely avoided
	nondeterminism = False

	def __init__(self, wactions, pactions):
		# Weights of the actions (dictionary)
		self.wactions = wactions
		# Fixed probabilities of the actions (dictionary)
		self.pactions = pactions

	@classmethod
	def parse(cls, text):
		# Parse the argument as a comma-separated dictionary
		arg_part = text[text.index('(')+1:-1] if '(' in text else ''
		kv_pairs = (map(str.strip, entry.split('=')) for entry in arg_part.split(',') if '=' in entry)

		wactions = {}
		pactions = {}

		for action, value in kv_pairs:
			# Separate the action name and the suffix (.p or .w)
			index = action.rfind('.')
			action, suffix = (action[:index], action[index:]) if index >= 0 else (action, '')

			try:
				# Keys ending in .p are fixed probabilities
				if suffix == '.p':
					p = float(value)

					if action in wactions:
						usermsgs.print_warning(f'Action {action} has already been assigned a weight. Ignoring its probability.')
					if 0 <= p <= 1:
						pactions[action] = p
					else:
						usermsgs.print_warning(f'Value {p} for action {action} is not a probability between 0 and 1.')

				# .w is optional when indicating weights
				else:
					if action in pactions:
						usermsgs.print_warning(f'Action {action} has already been assigned a probability. Ignoring its weight.')
					else:
						wactions[action] = float(value)

			# In case arguments cannot be parsed to float
			except ValueError:
				usermsgs.print_warning(f'Cannot parse value "{value}" for {action} in the weighted distribution.')

		return cls(wactions, pactions)

	def __call__(self, graph, state, successors):
		"""Assign the probability on the actions according to the weights"""

		# Actions leading to the successors
		# (this will never be called without successors, so actions is not empty)
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]

		# Calculate the frequency of the actions to distribute the probability uniformly
		# among their successors
		action_fq = Counter(actions)

		# Calculate the total probability and weight assigned
		assigned_p, total_w = 0, 0

		for action in action_fq.keys():
			p = self.pactions.get(action, None)

			if p is not None:
				assigned_p += p
			else:
				total_w += self.wactions.get(action, 1)

		# Assigns the fixed probabilities first and then distributes the rest among the
		# remaining actions according to their weights, if possible
		if 0 <= assigned_p <= 1 and (assigned_p > 0 or total_w > 0):
			# If assigned_p does not sum 1 and no weighted action is enabled
		  	# to distribute the rest, probabilities are scaled to 1
			p = 1 / assigned_p if total_w == 0 else (1 - assigned_p) / total_w

			action_fq = {action: self.pactions.get(action, p * self.wactions.get(action, 1)) / fq
			             for action, fq in action_fq.items()}

		# The total probability assigned surpass 1 or it is zero and no weight
		# is assigned to the other actions
		else:
			usermsgs.print_warning(f'Probabilities assigned to actions sum {assigned_p}. Ignoring.')

			# Uniform probabilities are assigned by default
			p = 1.0 / len(action_fq)
			action_fq = {action: p / fq for action, fq in action_fq.items()}

		return [action_fq[action] for action in actions]


def _find_vars(term, varset):
	"""Find all variables in a term"""

	if term.isVariable():
		varset.add(term)
	else:
		for arg in term.arguments():
			_find_vars(arg, varset)


class MaudeTermBaseAssigner:
	"""Assign probabilities according to the evaluation of a Maude term (base class)"""

	def __init__(self, term, replacements):
		self.term = term

		self.left_var = replacements['L']
		self.right_var = replacements['R']
		self.action_var = replacements['A']

		self.cache = {}

		self.module = term.symbol().getModule()
		qid_sort = self.module.findSort('Qid')
		self.qid_kind = qid_sort.kind() if qid_sort else None

	@classmethod
	def parse(cls, module, text):
		if '(' not in text:
			return None

		nat_sort = module.findSort('Nat')
		float_sort = module.findSort('Float')
		qid_sort = module.findSort('Qid')
		qid_kind = qid_sort.kind() if qid_sort else None

		# Parse the text between parentheses as a term
		symb_name = text[text.index('(')+1:-1]
		term = module.parseTerm(symb_name)

		if term is None:
			usermsgs.print_error(f'Cannot parse the weight function term {symb_name}.')
			return None

		# Check whether the result is a number
		sort = term.getSort()

		if (nat_sort is None or not (sort <= nat_sort)) and (float_sort is None or not (sort <= float_sort)):
			usermsgs.print_warning(f'The range sort of the term {symb_name} is not a valid number type. '
			                       'Continuing, but errors should be expected.')

		# Check the variables in the term (L is to be replaced by the left-hand
		# side, R by the right-hand side, and A by the transition label)
		variables = set()
		_find_vars(term, variables)
		repl = {'L': None, 'R': None, 'A': None}
		state_kind = None

		for v in variables:
			var_name = v.getVarName()

			if var_name in ('L', 'R'):
				var_kind = v.getSort().kind()

				# All ocurrences of the L and R variables in the term must belong to the same kind
				if state_kind:
					if var_kind != state_kind:
						usermsgs.print_warning(f'Variable {v} in weight assignment term belongs to a '
					                       f'wrong kind {var_kind} ({state_kind} expected).')
						return None
				else:
					state_kind = var_kind

				repl[var_name] = v

			elif var_name == 'A':
				if v.getSort().kind() != qid_kind:
					# This may be a false positive if the Qid sort is renamed
					usermsgs.print_warning(f'Variable {v} in weight assignment term belongs to a '
					                       f'wrong kind {v.getSort().kind()} ([Qid] expected).')
					return None

				repl[var_name] = v
			else:
				usermsgs.print_error(f'Unbound variable {v} in weight assignment term.')
				return None

		return cls(term, repl)

	def make_tuple(self, state, next_state, action):
		"""Make a tuple to serve as key for the cache"""

		key = []

		if self.left_var:
			key.append(state)
		if self.right_var:
			key.append(next_state)
		if self.action_var:
			key.append(action)

		return tuple(key)

	def get_weight(self, graph, term, state, next_state):
		"""Get the weight for the given transition"""

		# The action identifier (if needed)
		action = _get_stmt_label(graph, state, next_state) if self.action_var else None

		# Look for it in the cache to save work if possible
		w = self.cache.get(self.make_tuple(state, next_state, action), None)

		if w is None:
			# Prepare the term to be instantiated
			subs = {}

			# The left-hand side of the transition
			if self.left_var:
				subs[self.left_var] = term

			# The right-hand side of the transition
			if self.right_var:
				subs[self.right_var] = graph.getStateTerm(next_state)

			# The action identifier (will be parsed as Qid)
			if self.action_var:
				action_term = self.module.parseTerm(f"'{action if action is not None else 'unknown'}",
				                                    self.qid_kind)
				subs[self.action_var] = action_term

			# Build and reduce the term
			w = maude.Substitution(subs).instantiate(self.term)
			w.reduce()
			w = float(w)
			self.cache[self.make_tuple(state, next_state, action)] = w

		return w


class MaudeTermFullAssigner(MaudeTermBaseAssigner):
	"""Assign full probabilities to the successors by a Maude term"""

	# Nondeterminism is completely avoided
	nondeterminism = False

	def __init__(self, *args):
		super().__init__(*args)

	def __call__(self, graph, state, successors):

		term = graph.getStateTerm(state)

		weights = [self.get_weight(graph, term, state, next_state) for next_state in successors]
		total = sum(weights)

		# If all weights are null, we at least avoid dividing by zero
		if total == 0:
			usermsgs.print_warning(f'The weights for all the successors of {term} are null. Errors should be expected.')
			total = 1

		return [w / total for w in weights]


class MaudeTermMDPAssigner(MaudeTermBaseAssigner):
	"""Assign probabilities among successors for the same action as specified by a Maude term"""

	# This procedure yields Markov decision processes
	nondeterminism = True

	def __init__(self, *args):
		super().__init__(*args)

	def __call__(self, graph, state, successors):
		term = graph.getStateTerm(state)

		# Actions leading to the successors
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]
		# Weights for these successors
		weights = [self.get_weight(graph, term, state, next_state) for next_state in successors]

		return _assign_weights_by_actions(actions, weights, successors)


class UniformMDPAssigner:
	"""Assigner of probabilities uniformly on each nondeterministic action"""

	# This distribution yields a Markov decision process
	nondeterminism = True

	def __call__(self, graph, state, successors):
		# Actions leading to the successors
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]

		# Calculate the frequency of the actions to distribute the probability among them
		action_fq = Counter(actions)

		return [[(1 / action_fq[action], successors[k]) for k, a in enumerate(actions) if a == action]
	                for action in action_fq.keys()]


class RewardEvaluator:
	"""Evaluate reward terms on states"""

	def __init__(self, term, variable):
		self.reward_term = term
		self.variable = variable

	@staticmethod
	def new(term, state_kind=None):
		"""Construct a reward evaluator"""
		variables = set()
		_find_vars(term, variables)

		# If there is no variables, the reward is constant
		if not variables:
			term.reduce()
			constant = float(term)
			return lambda _: constant

		elif len(variables) > 1:
			usermsgs.print_warning('The reward term contains multiple variables '
			                       f'({", ".join(map(str, variables))}). It will be ignored.')
			return None

		# We check whether the variable is of the appropriate kind
		variable = variables.pop()

		if state_kind is not None and variable.getSort().kind() != state_kind:
			usermsgs.print_warning(f'The sort {variable.getSort()} of the reward term variable '
			                       'does not belong to the state kind. Errors should be expected.')

		return RewardEvaluator(term, variable)

	def __call__(self, state):
		term = maude.Substitution({self.variable: state}).instantiate(self.reward_term)
		term.reduce()

		# We do not warn if the term is not a Int or Float
		return float(term)


class AssignedGraph:
	"""Graph with probabilities by a probability assignment function"""

	def __init__(self, graph, assigner):
		self.graph = graph
		self.assigner = assigner
		self.visited = None

		self.nondeterminism = self.assigner.nondeterminism

	def getStateTerm(self, state):
		return self.graph.getStateTerm(state)

	def getNrRewrites(self):
		return self.graph.getNrRewrites()

	def getNrRealStates(self):
		return len(self.visited)

	def __len__(self):
		return self.graph.getNrStates()

	@property
	def strategyControlled(self):
		return self.graph.strategyControlled

	def getRule(self, *args):
		return self.graph.getRule(*args)

	def states(self):
		"""Iterator through the states (numbers) of the graph"""
		return self.visited

	def transitions(self):
		"""Iterator through the transitions of the graph"""

		for state in self.visited:
			successors = list(self.graph.getNextStates(state))

			if len(successors) == 1:
				yield state, ((1.0, successors[0]),)

			elif successors:
				probs = self.assigner(self.graph, state, successors)

				if self.assigner.nondeterminism:
					for action in probs:
						yield state, action
				else:
					yield state, zip(probs, successors)

	def expand(self):
		"""Expand the underlaying graph"""

		pending = [(0, self.graph.getNextStates(0))]
		self.visited = {0}

		while pending:
			# The stacks hold a state number and
			# an iterator to the children
			state, it = pending.pop()

			next_state = next(it, None)

			if next_state is not None:
				pending.append((state, it))

				if next_state not in self.visited:
					self.visited.add(next_state)
					pending.append((next_state, self.graph.getNextStates(next_state)))


class StrategyMarkovGraph:
	"""Graph with probabilities assigned by a strategy"""

	# This graph is always strategy-controlled
	strategyControlled = True

	def __init__(self, root):
		self.state_list = [root]
		self.state_map = {root: 0}
		self.nondeterminism = False

	def getStateTerm(self, state):
		return self.state_list[state].term

	def getNrRewrites(self):
		# The number of rewrites is not counted in this implementation
		return 0

	def getNrRealStates(self):
		return len(self.state_list)

	def __len__(self):
		return len(self.state_list)

	def states(self):
		"""Iterator through the states (numbers) of the graph"""
		return range(len(self.state_list))

	def transitions(self):
		"""Iterator through the transitions of the graph"""

		for state_nr, state in enumerate(self.state_list):

			for next_state in state.children:
				yield state_nr, ((1.0, self.state_map[next_state]),)

			for choice in state.child_choices:
				yield state_nr, ((p, self.state_map[next_state]) for next_state, p in choice.items())

	def expand(self):
		"""Expand the underlaying graph"""

		for k, state in enumerate(self.state_list):
			# Detect whether the graph contains unquantified nondeterminism
			if len(state.children) + len(state.child_choices) > 1:
				self.nondeterminism = True

			for child in chain(state.children, (c for choice in state.child_choices
				                              for c, _ in choice.items())):
				if child not in self.state_map:
					self.state_map[child] = len(self.state_list)
					self.state_list.append(child)


def get_probabilistic_strategy_graph(module, strategy, term):
	"""Get the probabilistic graph of a probabilistic strategy"""

	from . import pyslang

	ml = maude.getModule('META-LEVEL')

	# Initialize the strategy compiler and optimizer
	sc = pyslang.StratCompiler(module, ml, use_notify=True, ignore_one=True)

	# Compile the strategy
	p = sc.compile(ml.upStrategy(strategy))

	try:
		# Execute the strategy
		root = pyslang.MarkovRunner(p, term).run()

		graph = StrategyMarkovGraph(root)
		graph.expand()

		return graph

	except pyslang.BadProbStrategy as bps:
		usermsgs.print_error(bps)

	return None


def _get_assignment_method(data, spec, allow_file=False):
	"""Parse the probability assignment method"""

	distr, graph = None, None

	# If the argument to assign begins with @ we load it from a file
	if allow_file and spec.startswith('@'):
		try:
			with open(spec[1:]) as asfile:
				spec = asfile.read().strip()

		except OSError as oe:
			usermsgs.print_error(f'Cannot load the probability assignment method from file: {oe}.')

			return None, None

	if spec == 'strategy':
		if data.strategy is None:
			usermsgs.print_error('A strategy expression must be provided to use the strategy assignment method.')

		else:
			graph = get_probabilistic_strategy_graph(data.module, data.strategy, data.term)

	else:
		distr, found = get_local_assigner(data.module, spec)

		if distr is None and not found:
			usermsgs.print_error(f'Unknown probability assignment method {spec}.')

	return distr, graph


def get_probabilistic_graph(data, spec, allow_file=False, purge_fails='default', merge_states='default'):
	"""Get the probabilistic graph from the given problem data"""

	# Obtain the assignment method or the graph if it is strategy
	distr, graph = _get_assignment_method(data, spec, allow_file)

	if distr is None and graph is None:
		return None

	# Generate the graph for the local probability distributions
	if distr:
		graph = AssignedGraph(create_graph(
				data.term,
				data.strategy,
				opaque=data.opaque,
				full_matchrew=data.full_matchrew,
				purge_fails=purge_fails,
				merge_states=merge_states,
				logic='CTL'),
			distr)

		graph.expand()

	return graph
