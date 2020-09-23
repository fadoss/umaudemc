#
# μ-calculus model-checking algorithm and parity game solver
#
# The parity game is generated as described in
#
#   Julian C. Bradfield, Igor Walukiewicz. The mu-calculus and Model Checking.
#   Handbook of Model Checking. 2018. DOI: 10.1007/978-3-319-10575-8_26.
#
# with small variations. The parity game solver is a custom implementation of
# the Zielonka's algorithm:
#
#   Wieslaw Zielonka. Infinite Games on Finitely Coloured Graphs with Applications
#   to Automata on Infinite Trees. Theor. Comput. Sci. 200(1-2). 1998.
#   DOI: https://doi.org/10.1016/S0304-3975(98)00009-7.
#
# CTL model checking is also possible via this model checker using a
# straightforward translation to μ-calculus.
#

from ..common import maude
from ..wrappers import create_graph


class ParityGame:
	"""Parity game"""

	class ParityState:
		def __init__(self, player, priority):
			self.player = player
			self.priority = priority
			self.successors = set()

		def __repr__(self):
			return f'node(player={self.player}, priority={self.priority}, {self.successors})'

	def __init__(self):
		self.states = []
		self.max_prio = 0

	def add_state(self, player, priority):
		"""
		Add a new state to the game
		:param player: The player this state belongs to.
		:type player: int (either 0 or 1)
		:param priority: The state priority.
		:type priority: int
		:returns: The index of the new state in the game.
		:rtype: int
		"""
		index = len(self.states)
		self.states.append(self.ParityState(player, priority))
		if priority > self.max_prio:
			self.max_prio = priority
		return index

	def add_edge(self, origin, dest):
		"""Add an edge between to states of the game"""
		self.states[origin].successors.add(dest)

	def __len__(self):
		"""Size of the game (number of states)."""
		return len(self.states)

	def solve(self):
		"""Compute the winning sets of the two players using Zielonka's algorithm."""
		whole_game  = set(range(len(self)))

		# The deadlock states where the other player cannot move, and then
		# the states where these can be forcibly reached
		finite_win0 = {n for n, state in enumerate(self.states)
				 if state.player == 1 and len(state.successors) == 0}
		finite_win1 = {n for n, state in enumerate(self.states)
				 if state.player == 0 and len(state.successors) == 0}

		finite_win0 = self.attractor(0, whole_game, finite_win0)
		finite_win1 = self.attractor(1, whole_game, finite_win1)

		# The recursive Zielonka's algorithm implemented in solve_rec does not
		# take finite plays into account, which are handled here. The recursive
		# algorithm receives a a game without deadlock states.
		# (this decomposition is sound, because if player i can force the game
		# to reach a deadlock state for 1-i, player 1-i cannot force reaching a
		# deadlock state for i nor an infinite play)
		w0, w1 = self.solve_rec(whole_game - finite_win0 - finite_win1)

		# The winning states for each player are the union of those in which
		# they have an infinite and those in which they a finite winning strategy
		return w0.union(finite_win0), w1.union(finite_win1)

	def attractor(self, player, subgame, subset):
		"""Compute the attractor of a subset on a subgraph for a player"""

		# Iteratively extend subset with the states where it is sure that
		# the given player can reach the previous subset.

		while True:
			# reach is what will be added to the attractor at each step
			reach = set()

			for n in subgame:
				state = self.states[n]
				subsucc = state.successors.intersection(subgame)

				# As the given player can choose the next move in its states, its
				# states with at least one successor in the previous attractor
				# are included in the new attractor. Since it cannot choose in
				# the other player's states, they must have all its successor
				# in subset to be added.
				if (state.player == player and not subsucc.isdisjoint(subset) or
				    state.player != player and subsucc <= subset):
					reach.add(n)

			# Nothing new is added to the attractor, stop
			if reach <= subset:
				break
			else:
				subset.update(reach)

		return subset

	def solve_rec(self, subgame):
		"""Zielonka's algorithm for solving parity games"""

		p = max({self.states[n].priority for n in subgame}, default=0)

		# If the maximal priority is zero, player 0 wins from any state
		if p == 0:
			return subgame, set()
		else:
			# Player that wins with this priority
			i = p % 2
			# Nodes in subgame with priority p
			U = {n for n in subgame if self.states[n].priority == p}

			# States where player i can force the game to reach U (to win)
			A = self.attractor(i, subgame, U)

			# Solve the rest of the game
			W1 = self.solve_rec(subgame - A)

			# Player i wins in all the states of subgame - A,
			# so in all states of subgame too
			if W1[1-i] == set():
				return (subgame, set()) if i == 0 else (set(), subgame)

			# States where player i-1 can force the game to reach
			# its winning set in subgame - A
			B = self.attractor(1-i, subgame, W1[i-1])

			# Solve the remaining part of the game
			W2 = self.solve_rec(subgame - B)

			W2[1-i].update(B)

			return W2


class MuMaudeGame(ParityGame):
	"""Parity game that solve the μ-calculus for Maude models"""

	class FormulaNode:
		def __init__(self, head):
			self.children = []
			self.head     = head
			# The extra argument can be the variable name,
			# the set of labels or the atomic proposition term
			self.extra    = None
			self.adepth   = 0

		def __repr__(self):
			return f'node({self.head}, {self.extra}, {self.adepth}, {self.children})'

	def __init__(self, graph, formula):
		super(MuMaudeGame, self).__init__()

		# The module where the graph is defined
		self.module = graph.getStateTerm(0).symbol().getModule()
		self.graph = graph

		# Preprocess the formula to assign indices to its subformulae
		self.formula_nodes = []
		self.prepare_formula(formula)

		# Table of seen nodes
		self.seen = {}

		# Prepare itself to check atomic propositions
		bool_kind  = self.module.findSort('Bool').kind()
		state_kind = self.module.findSort('State').kind()
		prop_kind  = self.module.findSort('Prop').kind()

		self.true      = self.module.parseTerm('true', bool_kind)
		self.satisfies = self.module.findSymbol('_|=_', [state_kind, prop_kind], bool_kind)

		# Number of rewrites used to check this atomic proposition
		self.nrRewrites = 0

	def getNrRewrites(self):
		"""Total number of rewrites used to generate the game"""
		return self.nrRewrites + self.graph.getNrRewrites()

	# The following two functions are used to calculate the alternation
	# depth of the μ-calculus variables, which is used as the state
	# priority in the parity game.

	@staticmethod
	def merge_varinfo(one, other):
		"""Merge variable information from different sources"""

		# The variable information is a map from variable names
		# to pairs that contain the alternation depth of the
		# variable (in a subformula) and a set of alternation
		# flags (mu or nu), i.e. the types of the last fixpoint
		# operators that enclose a formula with that variable.

		# This function combines two dictionaries taking the
		# maximum when both have the same entry.

		for var, (depth, alter) in other.items():
			one_depth, one_alter = one.get(var, (-1, []))

			if depth > one_depth:
				one[var] = depth, alter
			elif depth == one_depth:
				one[var] = depth, alter.union(one_alter)

	@staticmethod
	def step_varinfo(one, kind):
		"""Increase the alternation depth if needed"""

		# Increase the alternation depth only if (one of) the last
		# fixpoint operator that make a variable reach its depth
		# is of a different kind.

		for var, (depth, alter) in one.items():
			if alter == {kind}:
				one[var] = (depth, {kind})
			else:
				one[var] = (depth + 1, {kind})

	def prepare_formula(self, formula, bindings={}):
		"""Prepare the given formula as a graph and calculate alternation depths"""

		head, *rest = formula

		# The formula is converted into a graph where each node
		# represent a subformula and is linked to its argument.

		# The bindings dictionary maps variable names to the index
		# of the subformula where they were bound. In the graph,
		# variables disappear and their parents are linked to those
		# subformulae, making a loop that unrolls the fixpoint.

		if head == 'Var':
			# Variables are linked to the term where they
			return bindings[rest[0]], {rest[0]: (0, set())}

		index = len(self.formula_nodes)
		vars  = {}
		node  = self.FormulaNode(head)
		self.formula_nodes.append(node)

		if head == 'Prop':
			node.extra = self.module.parseTerm(rest[0])

		elif head == '<_>_' or head == '`[_`]_':
			node.extra = rest[0]
			arg_index, vars = self.prepare_formula(rest[1], bindings)
			node.children.append(arg_index)

		elif head == 'mu_._' or head == 'nu_._':
			if rest[0][0] != 'Var':
				raise RuntimeError('Unexpected format for formula.')
			# The name of the variable
			node.extra = rest[0][1]
			# The new variable is added to the bindings
			# (a copy is needed because variable need not be fresh)
			arg_index, vars = self.prepare_formula(rest[1], dict(bindings, **{node.extra: index}))
			node.children.append(arg_index)
			self.step_varinfo(vars, head[:2])
			node.adepth = vars.get(node.extra, (0, 0))[0]

		else:
			for arg in rest:
				arg_index, arg_vars = self.prepare_formula(arg, bindings)
				self.merge_varinfo(vars, arg_vars)
				node.children.append(arg_index)

		return index, vars

	def check_aprop(self, state, aprop):
		term  = self.graph.getStateTerm(state)
		t = self.satisfies.makeTerm([term, aprop])
		self.nrRewrites += t.reduce()
		return t.equal(self.true)

	def match_label(self, state, next_state, labels):
		"""Check whether the given transition match the set of labels"""

		if self.graph.strategyControlled:
			transition = self.graph.getTransition(state, next_state)
			return transition.getType() == maude.StrategyRewriteGraph.RULE_APPLICATION and \
			       transition.getRule().getLabel() in labels
		else:
			return self.graph.getRule(state, next_state).getLabel() in labels

	def build_succs(self, state, formula, labels=None):
		"""Build the successor by rewriting for a given node"""

		return [self.build(next_state, formula)
			for next_state in self.graph.getNextStates(state)
			if labels is None or self.match_label(state, next_state, labels)]

	def build(self, state=0, fnode_nr=0):
		"""Build the parity game from a given state and node of the formula graph"""

		# Loops are detected with a table
		entry = self.seen.get((state, fnode_nr))

		if entry is not None:
			return entry
		else:
			self.seen[(state, fnode_nr)] = len(self)

		# The already processed subformula
		fnode = self.formula_nodes[fnode_nr]
		head  = fnode.head

		# Player 0 aims to prove the formula and player 1 to
		# refute it, so the nodes are assigned accordingly.
		# Only fixpoint nodes have non-zero priority.

		if head == 'Prop':
			player = 1 if self.check_aprop(state, fnode.extra) else 0
			return self.add_state(player, 0)

		elif head == '~_':
			# Formulae is assumed to be in negative normal form
			player = 0 if self.check_aprop(state, self.formula_nodes[fnode.children[0]].extra) else 1
			return self.add_state(player, 0)

		elif head == 'True':
			return self.add_state(1, 0)

		elif head == 'False':
			return self.add_state(0, 0)

		elif head == '_/\\_':
			parent = self.add_state(1, 0)
			left   = self.build(state, fnode.children[0])
			right  = self.build(state, fnode.children[1])
			self.add_edge(parent, left)
			self.add_edge(parent, right)
			return parent

		elif head == '_\\/_':
			parent = self.add_state(0, 0)
			left   = self.build(state, fnode.children[0])
			right  = self.build(state, fnode.children[1])
			self.add_edge(parent, left)
			self.add_edge(parent, right)
			return parent

		elif head == '`[.`]_':
			parent = self.add_state(1, 0)
			for succ in self.build_succs(state, fnode.children[0]):
				self.add_edge(parent, succ)
			return parent

		elif head == '<.>_':
			parent = self.add_state(0, 0)
			for succ in self.build_succs(state, fnode.children[0]):
				self.add_edge(parent, succ)
			return parent

		elif head == '`[_`]_':
			parent = self.add_state(1, 0)
			for succ in self.build_succs(state, fnode.children[0], labels=fnode.extra):
				self.add_edge(parent, succ)
			return parent

		elif head == '<_>_':
			parent = self.add_state(0, 0)
			for succ in self.build_succs(state, fnode.children[0], labels=fnode.extra):
				self.add_edge(parent, succ)
			return parent

		elif head == 'mu_._':
			parent = self.add_state(0, 2 * (fnode.adepth // 2) + 1)
			child  = self.build(state, fnode.children[0])
			self.add_edge(parent, child)
			return parent

		elif head == 'nu_._':
			parent = self.add_state(0, 2 * (fnode.adepth // 2))
			child  = self.build(state, fnode.children[0])
			self.add_edge(parent, child)
			return parent
		else:
			raise RuntimeError(f'Unexpected formula {head}.')

	def solve_mucalc(self):
		"""Generate a solve the μ-calulus game"""

		self.build()
		w0, w1 = self.solve()
		return 0 in w0


class BuiltinBackend:
	"""Backend for the builtin μ-calculus and CTL model checker"""

	def find(self):
		return True

	def ctl2mucalc(self, formula, universal=True, index=0):
		"""Translate a CTL formula to μ-calculus"""
		head, *rest = formula

		# This is an optimization that effectively reduces
		# the size of the translated formula
		if head == '_U_' and rest[0] == ['True']:
			head = '<>_'
			rest = [rest[1]]
		elif head == '_R_' and rest[0] == ['False']:
			head = '`[`]_'
			rest = [rest[1]]

		# Modality
		modop = '`[.`]_' if universal else '<.>_'

		if head == 'Prop':
			return formula
		elif head == 'A_':
			return self.ctl2mucalc(rest[0], True, index)
		elif head == 'E_':
			return self.ctl2mucalc(rest[0], False, index)
		elif head == 'O_':
			return [modop, self.ctl2mucalc(rest[0], index=index)]
		elif head == '_U_':
			return ['mu_._', ['Var', f'Z{index}'], ['_\\/_',
				  self.ctl2mucalc(rest[1], index=index + 1),
				  ['_/\\_', self.ctl2mucalc(rest[0], index=index + 1),
				  [modop, ['Var', f'Z{index}']]]]]
		elif head == '_R_':
			return ['nu_._', ['Var', f'Z{index}'], ['_/\\_',
				  self.ctl2mucalc(rest[1], index=index + 1),
				  ['_\\/_', self.ctl2mucalc(rest[0], index=index + 1),
				  [modop, ['Var', f'Z{index}']]]]]
		elif head == '`[`]_':
			return ['nu_._', ['Var', f'Z{index}'], ['_/\\_',
				  self.ctl2mucalc(rest[0], index=index + 1),
				  [modop, ['Var', f'Z{index}']]]]
		elif head == '<>_':
			return ['mu_._', ['Var', f'Z{index}'], ['_\\/_',
				  self.ctl2mucalc(rest[0], index=index + 1),
				  [modop, ['Var', f'Z{index}']]]]
		else:
			return [head] + [self.ctl2mucalc(arg, index=index) for arg in rest]

	def run(self, graph, formula, logic):
		"""Check a model-checking problem"""
		if logic == 'CTL':
			formula = self.ctl2mucalc(formula)

		game = MuMaudeGame(graph, formula)
		result = game.solve_mucalc()

		return result, {'states': graph.getNrStates(), 'rewrites': game.getNrRewrites(), 'game': len(game)}

	def check(self, graph=None, formula=None, logic=None, get_graph=False, **kwargs):
		"""Check a model-checking problem"""

		# Create the graph if not provided by the caller
		if graph is None:
			graph = create_graph(logic=logic, **kwargs)

		holds, stats = self.run(graph, formula, logic)

		if get_graph:
			stats['graph'] = graph

		return holds, stats
