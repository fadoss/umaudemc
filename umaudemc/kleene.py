#
# Strategy execution machine and graph for the Kleene star semantics
#

from .common import maude
from .pyslang import GraphRunner, GraphExecutionState


class KleeneExecutionState(GraphExecutionState):
	"""Execution state for generating a graph"""

	def __init__(self, term, pc, stack, conditional=False, graph_node=None, extra=None, iterations=None):
		super().__init__(term, pc, stack, conditional, graph_node,
		                 (extra, set() if iterations is None else iterations))

	def copy(self, term=None, pc=None, stack=None, conditional=False, graph_node=None, extra=None):
		"""Clone state with possibly some changes"""

		return KleeneExecutionState(
			self.term if term is None else term,
			self.pc + 1 if pc is None else pc,
			self.stack if stack is None else stack,
			conditional,
			self.graph_node if graph_node is None else graph_node,
			self.extra[0] if extra is None else extra,
			self.extra[1],
		)

	def add_kleene(self, args):
		"""Add iteration labels"""
		self.extra = (self.extra[0], self.extra[1] | {args})

	def reset_extra(self):
		"""Reset the extra attribute"""
		self.extra = (self.extra[0], set())


class KleeneRunner(GraphRunner):
	"""Runner that extract a Streett automaton with the Kleene star semantics"""

	def __init__(self, program, term):
		super().__init__(program, term, state_class=KleeneExecutionState)
		self.iter_contexts = {}

	def kleene(self, args, stack):
		"""Keep track of iterations"""
		context_id = self.iter_contexts.setdefault(self.current_state.stack, len(self.iter_contexts))
		self.current_state.add_kleene(((args[0], context_id), args[1]))
		super().kleene(args, stack)


class StrategyKleeneGraph:
	"""Strategy-controlled graph with iteration tracking"""

	# This is mostly copied from StrategyMarkovGraph
	# (merging some of the code should be considered)

	# This graph is always strategy-controlled
	strategyControlled = True

	class Transition:
		"""Transition of the StrategyKleeneGraph"""

		def __init__(self, pair=None):
			self.rule, self.iterations = pair

		def getRule(self):
			return self.rule

		def getType(self):
			return (maude.StrategyRewriteGraph.SOLUTION if self.rule is None
			        else maude.StrategyRewriteGraph.RULE_APPLICATION)

	def __init__(self, root, nrew):
		self.state_list = [root]
		self.state_map = {root: 0}
		self.nr_rewrites = nrew

	def getStateTerm(self, state):
		return self.state_list[state].term

	def getStateStrategy(self, state):
		return ''

	def getNrRewrites(self):
		return self.nr_rewrites

	def getNrRealStates(self):
		return len(self.state_list)

	def getNrStates(self):
		return len(self.state_list)

	def states(self):
		"""Iterator through the states (numbers) of the graph"""
		return range(len(self.state_list))

	def getTransition(self, origin, dest):
		"""Get one of the transitions from one state to another"""

		actions = self.state_list[origin].actions[self.state_list[dest]]
		return self.Transition(actions[0]) if actions else self.Transition()

	def getNextStates(self, stateNr):
		"""Iterator through the successors of a state"""

		for next_state in self.state_list[stateNr].children:
			yield self.state_map[next_state]

	def getTransitions(self, stateNr):
		"""Iterator through the transitions of the graph"""

		state = self.state_list[stateNr]
		for next_state in state.children:
			actions = state.actions[next_state]
			for action in (actions if actions else (None,)):
				yield self.state_map[next_state], self.Transition(action)

	def expand(self):
		"""Expand the underlying graph"""

		for k, state in enumerate(self.state_list):
			for child in state.children:
				if child not in self.state_map:
					self.state_map[child] = len(self.state_list)
					self.state_list.append(child)


def get_kleene_graph(data):
	"""Get the Kleene-aware graph for a strategy (with InputData)"""

	return get_kleene_graph_mts(data.module, data.term, data.strategy)


def get_kleene_graph_mts(module, term, strategy):
	"""Get the Kleene-aware graph for a strategy"""

	from . import pyslang

	ml = maude.getModule('META-LEVEL')

	# Initialize the strategy compiler and optimizer
	sc = pyslang.StratCompiler(module, ml, use_notify=True, use_kleene=True, ignore_one=True)

	# Compile the strategy
	p = sc.compile(ml.upStrategy(strategy))

	# Reduce the term
	nrew_initial = term.reduce()

	root, nrew = KleeneRunner(p, term).run()
	graph = StrategyKleeneGraph(root, nrew + nrew_initial)

	graph.expand()

	return graph
