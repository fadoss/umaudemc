#
# Wrappers around the Maude transition graphs
#
# They have the same interface as RewriteGraph and
# StrategyRewriteGraph with getNextState(state, index) replaced by a
# generator method getNextStates(state) that allows iterating over all the
# successors without indices (which are inconvenient in FailFreeGraph).
#

from .common import maude, default_model_settings


def default_getNextStates(self, stateNr):
	index = 0
	next_state = self.getNextState(stateNr, index)

	while next_state != -1:
		yield next_state
		index = index + 1
		next_state = self.getNextState(stateNr, index)


# Add getNextStates to the original graphs
maude.RewriteGraph.getNextStates = default_getNextStates
maude.StrategyRewriteGraph.getNextStates = default_getNextStates

# Add a property to know whether the graph is strategy-controlled
maude.RewriteGraph.strategyControlled = False
maude.StrategyRewriteGraph.strategyControlled = True


class WrappedGraph:
	"""Common methods for all wrapped graphs that do not change state numbering"""

	def __init__(self, graph):
		self.graph = graph

	def getStateTerm(self, stateNr):
		return self.graph.getStateTerm(stateNr)

	def getRule(self, *args):
		return self.graph.getRule(*args)

	def isSolutionState(self, stateNr):
		return self.graph.isSolutionState(stateNr)

	def getStateStrategy(self, stateNr):
		return self.graph.getStateStrategy(stateNr)

	def getTransition(self, *args):
		return self.graph.getTransition(*args)

	def getNrStates(self):
		return self.graph.getNrStates()

	def getNrRealStates(self):
		return self.graph.getNrRealStates()

	def getNrRewrites(self):
		return self.graph.getNrRewrites()

	def modelCheck(self, *args):
		return self.graph.modelCheck(*args)

	@property
	def strategyControlled(self):
		return self.graph.strategyControlled


class FailFreeGraph(WrappedGraph):
	"""A graph where failed states are removed"""

	def __init__(self, graph):
		super().__init__(graph)
		self.valid_states = [True]

	def expand(self, state=0):
		# Stack for the depth-first search
		# (state index, child index, whether the state is already valid)
		stack = [(state, 0, False)]
		# To pass whether the child is valid to the parent
		valid_child = False

		while stack:
			state, index, valid = stack.pop()

			next_state = self.graph.getNextState(state, index)

			# If we come back from a valid child
			if valid_child:
				valid, valid_child = True, False

			# No more successors
			if next_state == -1:
				# If no valid successor has been reached, the
				# state is not valid unless it is a solution
				valid_child = valid or self.graph.isSolutionState(state)
				self.valid_states[state] = valid_child

			# A new state, process it
			elif next_state >= len(self.valid_states):
				self.valid_states.append(True)
				stack.append((state, index + 1, valid))
				stack.append((next_state, 0, False))

			# The child is a known state
			# (if it is valid or in the path, then this state is valid)
			else:
				stack.append((state, index + 1, valid or self.valid_states[next_state]))

	def getNextStates(self, stateNr):
		for next_state in self.graph.getNextStates(stateNr):
			if self.valid_states[next_state]:
				yield next_state


class MergedGraph:
	"""A graph where states are merged"""

	def __init__(self, graph):
		self.graph = graph
		self.states = [frozenset({0})]
		self.table = {frozenset({0}): 0}

	def getStateTerm(self, stateNr):
		index = next(iter(self.states[stateNr]))
		return self.graph.getStateTerm(index)

	def isSolutionState(self, stateNr):
		return any(map(self.graph.isSolutionState, self.states[stateNr]))

	def getStateStrategy(self, stateNr):
		index = next(iter(self.states[stateNr]))
		return self.graph.getStateStrategy(index)

	def getTransition(self, origin, dest):
		for idx_origin in self.states[origin]:
			for idx_dest in self.states[dest]:
				trans = self.graph.getTransition(idx_origin, idx_dest)
				if trans is not None:
					return trans
		return None

	def getNrStates(self):
		return len(self.states)

	def getNrRealStates(self):
		return self.graph.getNrRealStates()

	def getNrRewrites(self):
		return self.graph.getNrRewrites()

	@property
	def strategyControlled(self):
		return self.graph.strategyControlled


class StateMergedGraph(MergedGraph):
	"""Graph where successors with a common underlying term are merged"""

	def getNextStates(self, stateNr):
		next_table = {}

		# next_table maps terms to the successor in stateNr
		# whose underlying term is the key

		for state in self.states[stateNr]:
			for next_state in self.graph.getNextStates(state):
				# Access to the internal state number could be useful here
				next_state_id = str(self.graph.getStateTerm(next_state))
				succ_set = next_table.setdefault(next_state_id, set())

				succ_set.add(next_state)

		# Then, the sets in next_table are the successors of the merged
		# state, and they are added to the graph if not already included

		for term, states in next_table.items():
			frozen_key = frozenset(states)
			index = self.table.get(frozen_key, None)

			if index is None:
				index = len(self.states)
				self.table[frozen_key] = index
				self.states.append(frozen_key)

			yield index


class EdgeMergedGraph(MergedGraph):
	"""Graph where successors with a common underlying term by a common action are merged"""

	@staticmethod
	def transition_id(transition):
		"""Identify a transition by its type and label"""
		tr_type = transition.getType()
		tr_label = 0

		if tr_type == maude.StrategyRewriteGraph.RULE_APPLICATION:
			tr_label = transition.getRule().getLabel()
		elif tr_type == maude.StrategyRewriteGraph.OPAQUE_STRATEGY:
			tr_label = transition.getStrategy().getName()

		return tr_type, tr_label

	def getNextStates(self, stateNr):
		next_table = {}

		# next_table maps term and action identifiers to the successors
		# in stateNr whose underlying term and generating transition are those

		for state in self.states[stateNr]:
			for next_state in self.graph.getNextStates(state):
				next_state_id = str(self.graph.getStateTerm(next_state))
				transition_id = EdgeMergedGraph.transition_id(self.graph.getTransition(state, next_state))
				succ_set = next_table.setdefault((transition_id, next_state_id), set())

				succ_set.add(next_state)

		# Then, the states are added to the graph

		for (action, term), states in next_table.items():
			frozen_key = frozenset(states)
			index = self.table.get((action, frozen_key), None)

			if index is None:
				index = len(self.states)
				self.table[(action, frozen_key)] = index
				self.states.append(frozen_key)

			yield index


class BoundedGraph(WrappedGraph):
	"""Graph of bounded depth"""

	def __init__(self, graph, depth):
		super().__init__(graph)
		self.depth = depth
		# Assigns depth to state numbers
		self.state_depth = {0: 0}

	def getNextStates(self, stateNr):
		this_depth = self.state_depth[stateNr]
		if this_depth < self.depth:
			for next_state in self.graph.getNextStates(stateNr):
				next_depth = self.state_depth.get(next_state)

				if next_depth is None or next_depth > this_depth + 1:
					self.state_depth[next_state] = this_depth + 1

				yield next_state


def wrapGraph(graph, purge_fails, merge_states):
	"""Wrap graphs according to the model modification options"""
	if purge_fails == 'yes':
		graph = FailFreeGraph(graph)
		graph.expand()
	if merge_states == 'state':
		graph = StateMergedGraph(graph)
	elif merge_states == 'edge':
		graph = EdgeMergedGraph(graph)

	return graph


def create_graph(term=None, strategy=None, opaque=(), full_matchrew=False, purge_fails='default',
                 merge_states='default', tableau=False, logic=None, **_):
	"""Create a graph from the problem input data"""

	if strategy is None:
		graph = maude.RewriteGraph(term)
	else:
		graph = maude.StrategyRewriteGraph(term, strategy, opaque, full_matchrew)

	# Wrap the graph with the model modification for branching-time
	purge_fails, merge_states = default_model_settings(logic, purge_fails, merge_states, strategy,
	                                                   tableau=tableau)
	return wrapGraph(graph, purge_fails, merge_states)
