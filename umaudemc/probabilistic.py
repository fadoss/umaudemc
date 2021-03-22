#
# Probabilistic model-checking support
#
# Many "probability assigners" are defined to distribute probabilities among the
# successors of each state in the model. Their "nondeterminism" property tells
# whether they yield MDPs or DTMCs. In the first case, the result of these
# assigner is a list of lists of (state id, p) tuples where each sublist
# represents a nondeterministic action. In the second case, the result is
# a single list with a probability number for each successor.
#

from collections import Counter

from .common import usermsgs, maude


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


def get_assigner(module, name):
	"""Instantiate probability assignment methods by their text descriptions"""

	if name == 'metadata':
		return MetadataAssigner(), True

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
	"""Uniform assignement of probabilities among successors"""

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
	mdata = None if stmt is None else stmt.getMetadata()

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
	# Grouped weights
	grouped = {action: [] for action in actions}

	for k, w in enumerate(weights):
		grouped[actions[k]].append((successors[k], w))

	# Normalize weights to probabilities by groups
	for gr in grouped.values():
		total = sum((w for _, w in gr))
		for k, (s, w) in enumerate(gr):
			gr[k] = s, (w / total)

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
	""""Weighted assignement of probabilisties among actions and then uniform among successors"""

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
			try:
				# Keys ending in .p are fixed probabilities
				if action.endswith('.p'):
					p = float(value)

					if action[:-2] in wactions:
						usermsgs.print_warning(f'Action {action[:-2]} has already been assigned a weight. Ignoring its probability.')
					if 0 <= p <= 1:
						pactions[action[:-2]] = p
					else:
						usermsgs.print_warning(f'Value {p} for action {action[:-2]} is not a probability between 0 and 1.')

				else:
					# .w is optional when indicating weights
					if action.endswith('.w'):
						action = action[:-2]

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

		# Calculate the assigned probability and the number of weighted elements
		# (weights are not affected by the multiplicity of each label)
		assigned_p, total_w = 0, 0

		for action in action_fq.keys():
			p = self.pactions.get(action, None)

			if p is not None:
				assigned_p += p
			else:
				total_w += self.wactions.get(action, 1)

		# Distribute the probability by first assigning the fixed probabilities and then
		# sharing the rest among the rest of actions according to their weights, if possible
		if 0 <= assigned_p <= 1 and (assigned_p > 0 or total_w > 0):
			# If assigned_p does not sum 1 but there is no other enabled actions
		  	# to distribute the rest, probabilities are scaled to 1
			p = 1 / assigned_p if total_w == 0 else (1 - assigned_p) / total_w

			action_fq = {action: self.pactions.get(action, p * self.wactions.get(action, 1)) / fq
			             for action, fq in action_fq.items()}

		# The total probability assigned to the available actions surpass 1 or it is
		# zero and no weight is assigned to the other actions
		else:
			usermsgs.print_warning(f'Probabilities assigned to actions sum {assigned_p}. Ignoring.')

			# Uniform probabilities are assigned by default
			p = 1.0 / len(action_fq)
			action_fq = {action: p / fq for action, fq in action_fq.items()}

		return [action_fq[action] for action in actions]


class MaudeTermBaseAssigner:
	"""Assign probabilities according to the evaluation of a Maude term (base class)"""

	def __init__(self, symb, args, replacements):
		self.symb = symb
		self.args = args
		self.repls = replacements

		self.cache = {}

		self.module = symb.getModule()
		self.qid_kind = self.module.findSort('Qid').kind()

	@classmethod
	def parse(cls, module, text):
		if '(' not in text:
			return None

		nat_sort = module.findSort('Nat')
		float_sort = module.findSort('Float')
		qid_kind = module.findSort('Qid').kind()
		state_kind = module.findSort('State').kind()

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

		# Process the arguments to be replaced later (L is to be replaced by the
		# left-hand side, R by the right-hand side, and A by the transition label)
		args = tuple(term.arguments())
		repl = {'L': [], 'R': [], 'A': []}

		for k, a in enumerate(args):
			var_name = a.getVarName()

			if a.getSort().kind() == state_kind:
				if var_name in ('L', 'R'):
					repl[var_name].append(k)

			elif a.getSort().kind() == qid_kind:
				if var_name == 'A':
					repl['A'].append(k)

		return cls(term.symbol(), args, repl)

	def get_weight(self, graph, term, state, next_state):
		"""Get the weight for the given transition"""

		# Look for it in the cache to save work if possible
		w = self.cache.get((state, next_state), None)

		if w is None:
			# Prepare the term to be reduced
			args = list(self.args)

			# The left-hand side of the transition
			for k in self.repls['L']:
				args[k] = term

			# The right-hand side of the transition
			if self.repls['R']:
				next_term = graph.getStateTerm(next_state)
				for k in self.repls['R']:
					args[k] = next_term

			# The action identifier (will be parsed as Qid)
			if self.repls['A']:
				action = _get_stmt_label(graph, state, next_state)
				action_term = self.module.parseTerm(f"'{action if action is not None else 'unknown'}",
				                                    self.qid_kind)
				for k in self.repls['A']:
					args[k] = action_term

			# Build and reduce the term
			w = self.symb.makeTerm(args)
			w.reduce()
			w = float(w)
			self.cache[(state, next_state)] = w

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

		return [[(successors[k], 1 / action_fq[action]) for k, a in enumerate(actions) if a == action]
	                for action in action_fq.keys()]


def _dissect_term(term, variable):
	"""Dissect a term"""

	# None indicates that the variable value must be filled there
	if term.getVarName() == variable:
		return None

	# Composed terms are transformed to a list whose head is the symbol and with
	# its arguments in the tail. However, if term does not contain the variable,
	# it is keep as is
	else:
		dissected_args = [_dissect_term(arg, variable) for arg in term.arguments()]
		contains_var = any(map(lambda v: not isinstance(v, maude.Term), dissected_args))

		return [term.symbol()] + dissected_args if contains_var else term


def _recompose_term(dterm, value):
	"""Recompose terms dissected by _dissect_term while instantiating its variable"""

	if dterm is None:
		return value

	elif isinstance(dterm, list):
		symb, *args = dterm
		return symb.makeTerm([_recompose_term(arg, value) for arg in args])

	else:
		return dterm


class RewardEvaluator:
	"""Evaluate reward terms on states"""

	def __init__(self, term):
		# Substitutions should be used instead of this when they
		# are properly supported by the maude library
		self.dissected_term = _dissect_term(term, 'S')

	def __call__(self, state):
		term = _recompose_term(self.dissected_term, state)
		term.reduce()

		# We do not warn if the term is not a Int or Float
		return float(term)
