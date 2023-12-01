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


def get_local_assigner(module, name, stmt_weights=None):
	"""Instantiate probability assignment methods by their text descriptions"""

	# Assignments for CTMC are the same as assignments for DTMC,
	# but the caller should not normalize the weights
	if name.startswith('ctmc-'):
		name = name[5:]

		# ctmc- and mdp- cannot be combined
		if name.startswith('mdp-'):
			return None, False

	if name == 'metadata':
		return MetadataAssigner(stmt_weights), True

	elif name == 'mdp-metadata':
		return MetadataMDPAssigner(stmt_weights), True

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
		return [1.0] * degree


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


def _get_metadata(graph, origin, dest, stmt_weights):
	"""Parse the metadata for the given transition"""

	# Get the statement producing the transition
	stmt = _get_transition_stmt(graph, origin, dest)

	return stmt_weights.get(stmt, 1.0)


def parse_metadata_weights(module):
	"""Get a dictionary with the weights in the metadata of all statements in the module"""

	meta_dict, num_errors, has_term = {}, 0, False

	# Parse metadata from rules and strategies
	for stmt in chain(module.getRules(), module.getStrategies()):
		mdata = stmt.getMetadata()

		if mdata:
			try: meta_dict[stmt] = float(mdata)
			except ValueError:
				# Generalized metadata is not supported for strategies
				if isinstance(stmt, maude.RewriteStrategy):
					continue

				# Get the variables in the rule to allow using
				# them in the metadata attribute
				bound_vars = set()
				_find_bound_vars(stmt, bound_vars)

				term = module.parseTerm(mdata, None, list(bound_vars))

				# The attribute can be parsed as a term
				if term:
					# Check whether all variables are bound
					attribute_vars = set()
					_find_vars(term, attribute_vars)

					if not attribute_vars <= bound_vars:
						example, *_ = attribute_vars - bound_vars
						usermsgs.print_warning(f'Unbound variable "{example}" in the metadata attribute '
						                       f'of rule at {stmt.getLineNumber()}. Assuming weight 1.')
						num_errors += 1
					else:
						meta_dict[stmt] = term
						has_term = True
				else:
					usermsgs.print_warning('Cannot parse as a term the metadata attribute '
					                       f'of rule at {stmt.getLineNumber()}. Assuming weight 1.')
					num_errors += 1

	return meta_dict, num_errors, has_term


class MetadataAssigner:
	"""Distribute the weights among the successors according
	   to the annotations in the metadata attribute"""

	# Nondeterminism is completely avoided
	nondeterminism = False

	def __init__(self, stmt_weights):
		self.stmt_weights = stmt_weights

	def __call__(self, graph, state, successors):
		# Get the weights from the metadata attributes (assuming 1 by default)
		return [_get_metadata(graph, state, next_state, self.stmt_weights)
		        for next_state in successors]


def _get_stmt_label(graph, origin, dest):
	"""Get the label of a statement"""

	stmt = _get_transition_stmt(graph, origin, dest)

	if stmt is None:
		return stmt

	return stmt.getLabel() if hasattr(stmt, 'getLabel') else stmt.getName()


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

	def __init__(self, stmt_weights):
		self.stmt_weights = stmt_weights

	def __call__(self, graph, state, successors):
		# Actions of the successors
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]
		# Weights of the successors (1 by default)
		weights = [_get_metadata(graph, state, next_state, self.stmt_weights) for next_state in successors]

		return _assign_weights_by_actions(actions, weights, successors)


class WeightedActionAssigner:
	""""Weighted selection among actions and then uniform selection among successors"""

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
		"""Assign the weights or probabilities of the actions"""

		# Actions leading to the successors
		# (this will never be called without successors, so actions is not empty)
		actions = [_get_stmt_label(graph, state, next_state) for next_state in successors]

		# Calculate the frequency of the actions to distribute the probability uniformly
		# among their successors
		action_fq = Counter(actions)

		# Calculate the total probability and weight assigned
		assigned_p, total_w = 0.0, 0

		for action in action_fq.keys():
			p = self.pactions.get(action, None)

			if p is not None:
				assigned_p += p
			else:
				total_w += self.wactions.get(action, 1)

		# Assigns the fixed probabilities first and then distributes the rest among the
		# remaining actions according to their weights, if possible
		if 0.0 < assigned_p <= 1.0:
			# If assigned_p does not sum 1 and no weighted action is enabled
		  	# to distribute the rest, probabilities are scaled to 1
			p = 1 / assigned_p if total_w == 0 else (1.0 - assigned_p) / total_w

			action_fq = {action: self.pactions.get(action, p * self.wactions.get(action, 1.0)) / fq
			             for action, fq in action_fq.items()}

		# There are no fixed probabilities (or they are not valid), so we distribute the
		# probabilities among the actions by their weights without normalizing
		else:
			# The total probability assigned surpass 1, we ignore fixed probabilities
			if assigned_p > 1.0:
				usermsgs.print_warning(f'Probabilities assigned to actions sum {assigned_p}. Ignoring.')

			# Uniform probabilities are assigned by default
			action_fq = {action: self.wactions.get(action, 1.0) / fq for action, fq in action_fq.items()}

		return [action_fq[action] for action in actions]


def _find_vars(term, varset):
	"""Find all variables in a term"""

	if term.isVariable():
		varset.add(term)
	else:
		for arg in term.arguments():
			_find_vars(arg, varset)


def _find_bound_vars(stmt, varset):
	"""Find all bound variables in a rule (or equation)"""

	_find_vars(stmt.getLhs(), varset)

	for fragment in stmt.getCondition():
		if isinstance(fragment, maude.AssignmentCondition):
			_find_vars(fragment.getLhs(), varset)
		elif isinstance(fragment, maude.RewriteCondition):
			_find_vars(fragment.getRhs(), varset)


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

				# All occurrences of the L and R variables in the term must belong to the same kind
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

		return [self.get_weight(graph, term, state, next_state)
		        for next_state in successors]


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

		return [[(1.0 / action_fq[action], successors[k]) for k, a in enumerate(actions) if a == action]
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

		# We do not warn if the term is not an Int or Float
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
		"""Expand the underlying graph"""

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


class GeneralizedMetadataGraph:
	"""Graph with probabilities given by non-ground metadata"""

	strategyControlled = False

	def __init__(self, term, stmt_map, mdp=False):
		self.terms = [term]
		self.term_map = {term: 0}
		self.edges = []
		self.rewrite_count = term.reduce()
		self.stmt_map = stmt_map

		self.nondeterminism = mdp

	def getStateTerm(self, state):
		return self.terms[state]

	def getNrRewrites(self):
		return self.rewrite_count

	def getNrRealStates(self):
		return len(self.terms)

	def __len__(self):
		return len(self.terms)

	def getRule(self, origin, dest):
		return next((rl for succ, rl, _ in self.edges[origin] if succ == dest), None)

	def states(self):
		"""Iterator through the states (numbers) of the graph"""
		return range(len(self.terms))

	def transitions(self):
		"""Iterator through the transitions of the graph"""

		for state, edges in enumerate(self.edges):
			if len(edges) == 1:
				next_state, _, w = edges[0]
				yield state, ((w, next_state),)

			elif edges:
				if self.nondeterminism:
					successors, actions, weights = zip(*edges)
					actions = [rl.getLabel() for rl in actions]

					for action in _assign_weights_by_actions(actions, weights, successors):
						yield state, action
				else:
					yield state, ((w, next_state) for next_state, _, w in edges)

	def expand(self):
		"""Expand the underlying graph"""

		# States are numbered from zero
		state, num_states = 0, 1

		while state < num_states:
			term = self.terms[state]

			# Edges are rewrites obtained with Term.apply
			edges = []

			for child, subs, ctx, rl in term.apply(None):
				# Term.apply do not reduce the terms itself
				self.rewrite_count += 1 + child.reduce()
				index = self.term_map.get(child)

				# A new term, so a new state
				if index is None:
					index = len(self.terms)
					self.terms.append(child)
					self.term_map[child] = index
					num_states += 1

				# Evaluate the metadata weight
				weight = self.stmt_map.get(rl, 1.0)

				# If it is not a literal, we should reduce the term
				if not isinstance(weight, float):
					weight = subs.instantiate(weight)
					self.rewrite_count += weight.reduce()
					weight = float(weight)

				edges.append((index, rl, weight))

			self.edges.append(edges)
			state += 1


class StrategyMarkovGraph:
	"""Graph with weights assigned by a strategy"""

	# This graph is always strategy-controlled
	strategyControlled = True

	class Transition:
		"""Transition of the StrategyMarkovGraph"""

		def __init__(self, rule=None):
			self.rule = rule

		def getRule(self):
			return self.rule

		def getType(self):
			return (maude.StrategyRewriteGraph.SOLUTION if self.rule is None
			        else maude.StrategyRewriteGraph.RULE_APPLICATION)

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

	def getTransition(self, origin, dest):
		actions = self.state_list[origin].actions[self.state_list[dest]]
		return self.Transition(actions[0]) if actions else self.Transition()

	def states(self):
		"""Iterator through the states (numbers) of the graph"""
		return range(len(self.state_list))

	def transitions(self):
		"""Iterator through the transitions of the graph"""

		for state_nr, state in enumerate(self.state_list):

			for next_state in state.children:
				yield state_nr, ((1.0, self.state_map[next_state]),)

			for choice in state.child_choices:
				# Normalize weights in the case of an MDP
				if self.nondeterminism:
					total_w = sum(choice.values())
					yield state_nr, ((w / total_w, self.state_map[next_state]) for next_state, w in choice.items())
				else:
					yield state_nr, ((w, self.state_map[next_state]) for next_state, w in choice.items())

	def expand(self):
		"""Expand the underlying graph"""

		for k, state in enumerate(self.state_list):
			# Detect whether the graph contains unquantified nondeterminism
			if len(state.children) + len(state.child_choices) > 1:
				self.nondeterminism = True

			for child in chain(state.children, (c for choice in state.child_choices
			                                    for c, _ in choice.items())):
				if child not in self.state_map:
					self.state_map[child] = len(self.state_list)
					self.state_list.append(child)


class StrategyMetadataGraph(StrategyMarkovGraph):
	"""Strategic graph with probabilities given by non-ground metadata"""

	def __init__(self, root, mdp=False):
		super().__init__(root)
		self.nondeterminism = mdp

	def getTransition(self, origin, dest):
		actions = self.state_list[origin].actions[self.state_list[dest]]
		return self.Transition(actions[0][0]) if actions else self.Transition()

	def transitions(self):
		"""Iterator through the transitions of the graph"""

		# Strategies may be allowed to stop or continue when a solution
		# is found, but stopping does not have a weight assigned by the
		# metadata method, so we assume 1 as for unassigned rules
		default_action = ((None, 1.0),)

		for state_nr, state in enumerate(self.state_list):
			action_map = state.actions

			if self.nondeterminism:
				successors = {}

				# Group weights and successors by rule label
				for target, rls in action_map.items():
					target_nr = self.state_map[target]
					for rl, w in rls:
						label = rl.getLabel() if rl else None
						succ4action = successors.setdefault(label, {})
						succ4action[target_nr] = succ4action.get(target_nr, 0.0) + w

				# Yield the distribution of successors for each action (normalized)
				for action, succs in successors.items():
					total_w = sum(succs.values())
					yield state_nr, ((w / total_w, target) for target, w in succs.items())

			else:
				yield state_nr, ((sum(w for _, w in (action_map.get(next_state) or default_action)),
				                  self.state_map[next_state]) for next_state in state.children)

	def expand(self):
		"""Expand the underlying graph"""

		for k, state in enumerate(self.state_list):
			# state.child_choices must be empty because
			# MetadataRunner does not introduce choices

			for child in chain(state.children):
				if child not in self.state_map:
					self.state_map[child] = len(self.state_list)
					self.state_list.append(child)


def get_probabilistic_strategy_graph(module, strategy, term, stmt_map=None, mdp=False):
	"""Get the probabilistic graph of a probabilistic strategy"""

	from . import pyslang

	ml = maude.getModule('META-LEVEL')

	# Initialize the strategy compiler and optimizer
	sc = pyslang.StratCompiler(module, ml, use_notify=True, ignore_one=True)

	# Compile the strategy
	p = sc.compile(ml.upStrategy(strategy))

	try:
		# Reduce the term
		term.reduce()

		# Execute the strategy with the corresponding runner
		if stmt_map is None:
			root = pyslang.MarkovRunner(p, term).run()
			graph = StrategyMarkovGraph(root)
		else:
			root = pyslang.MetadataRunner(p, term, stmt_map).run()
			graph = StrategyMetadataGraph(root, mdp)

		graph.expand()

		return graph

	except pyslang.BadProbStrategy as bps:
		usermsgs.print_error(bps)

	return None


def _get_assignment_method(data, spec, allow_file=False):
	"""Parse the probability assignment method"""

	# If the argument to assign begins with @ we load it from a file
	if allow_file and spec.startswith('@'):
		try:
			with open(spec[1:]) as asfile:
				spec = asfile.read().strip()

		except OSError as oe:
			usermsgs.print_error(f'Cannot load the probability assignment method from file: {oe}.')

			return None, None

	# Special cases: the general case explained below cannot be applied for some assignment methods
	# since the graphs obtained from the Maude library do not provide enough information

	# The 'strategy' method uses a custom Python-based graph, since StrategyRewriteGraph
	# in the Maude library chooses one of the probabilistic children at random
	if spec in ('strategy', 'ctmc-strategy'):
		if data.strategy is None:
			usermsgs.print_error('A strategy expression must be provided to use the strategy assignment method.')

		else:
			return None, get_probabilistic_strategy_graph(data.module, data.strategy, data.term)

	# The 'metadata' attribute with non-ground weights also uses a custom graph, since the matching
	# substitution of a transition cannot be obtained from the graph provided by the library
	stmt_weights = None

	if spec.endswith('metadata'):
		stmt_weights, num_errors, has_term = parse_metadata_weights(data.module)

		if has_term:
			if data.strategy:
				graph = get_probabilistic_strategy_graph(data.module, data.strategy, data.term,
				                                         stmt_map=stmt_weights,
				                                         mdp=spec == 'mdp-metadata')
			else:
				graph = GeneralizedMetadataGraph(data.term, stmt_weights, mdp=spec == 'mdp-metadata')

			graph.expand()
			return None, graph

	# General case: a local assignment function is used to assign probabilities
	# to the children of every node in the graphs provided by the Maude library

	distr, found = get_local_assigner(data.module, spec, stmt_weights)

	if distr is None and not found:
		usermsgs.print_error(f'Unknown probability assignment method {spec}.')

	return distr, None


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
