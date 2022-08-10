#
# Spot backend for umaudemc
#

import time

import buddy
import spot

from ..formulae import collect_aprops
from ..opsem import OpSemKleeneInstance, OpSemGraph
from ..wrappers import create_graph


class SpotModelBuilder:
	"""Base class for building models for Spot"""

	def __init__(self, graph, aprops, bdd_dict):
		self.graph = graph
		self.aprops = aprops
		self.bdd_dict = bdd_dict

		# BDD variables for each atomic proposition
		self.bdd_vars = None
		# Mapping from Maude states to Spot states
		self.state_map = {}
		# Number of rewrites due to atomic proposition evaluation
		self.nrRewrites = 0

		# Model-checking stuff
		self.true = None
		self.satisfies = None

	def _init_model_checking(self, module):
		"""Init the model-checking stuff"""

		# Find the satisfaction (|=) symbol and the true constant to be used
		# when testing atomic propositions.

		bool_kind = module.findSort('Bool').kind()
		state_kind = module.findSort('State').kind()
		prop_kind = module.findSort('Prop').kind()

		self.true = module.parseTerm('true', bool_kind)
		self.satisfies = module.findSymbol('_|=_', (state_kind, prop_kind), bool_kind)

	def _register_aprops(self, automaton):
		"""Register the atomic proposition in the automaton"""

		self.bdd_vars = [buddy.bdd_ithvar(automaton.register_ap(str(prop))) for prop in self.aprops]

	def _state_label(self, term):
		"""Evaluate the atomic propositions in a term of the graph and return a BDD"""

		bdd_value = buddy.bddtrue

		for i, prop in enumerate(self.bdd_vars):
			test = self.satisfies(term, self.aprops[i])
			self.nrRewrites += test.reduce()
			bdd_value &= prop if test == self.true else - prop

		return bdd_value

	def extract_counterexample(self, run, automaton):
		"""Extract the counterexample from an automaton run"""

		# Project the part of the product run for the model automaton
		prun = run.project(automaton)

		# Map from Spot indices to Maude indices
		state_inv = {v: k for k, v in self.state_map.items()}

		prefix = [state_inv[automaton.state_number(step.s)] for step in prun.prefix]
		cycle = [state_inv[automaton.state_number(step.s)] for step in prun.cycle]

		# Spot includes the start of the loop twice, unlike Maude
		if prefix and cycle and prefix[-1] == cycle[0]:
			prefix.pop()

		return prefix, cycle

	def getNrRewrites(self):
		return self.nrRewrites + self.graph.getNrRewrites()


class StandardModelBuilder(SpotModelBuilder):
	""""Generator of Kripke structure for standard Maude models"""

	def __init__(self, graph, *args):
		super().__init__(graph, *args)

		# Init the model checking stuff
		module = graph.getStateTerm(0).symbol().getModule()
		self._init_model_checking(module)

	def build(self):
		"""Create a Kripke structure for the rewrite graph"""

		# A Kripke graph (in C++ we could have defined this Kripke
		# structure on the fly, but this is not available in Python)
		k = spot.make_kripke_graph(self.bdd_dict)

		# Register and get the BDD variables for the atomic propositions
		self._register_aprops(k)

		# Depth-first exploration of the graph while creating the Kripke structure
		pending = [0]
		self.state_map = {0: k.new_state(self._state_label(self.graph.getStateTerm(0)))}

		while pending:
			state = pending.pop()
			state_spot = self.state_map[state]
			deadlock = True

			for next_state in self.graph.getNextStates(state):
				next_state_spot = self.state_map.get(next_state)

				if next_state_spot is None:
					next_state_spot = k.new_state(self._state_label(self.graph.getStateTerm(next_state)))
					self.state_map[next_state] = next_state_spot
					pending.append(next_state)

				k.new_edge(state_spot, next_state_spot)
				deadlock = False

			# Add a self-loop to deadlock states
			if deadlock and not self.graph.strategyControlled:
				k.new_edge(state_spot, state_spot)

		k.set_init_state(self.state_map[0])

		return k


class KleeneModelBuilder(SpotModelBuilder):
	"""Generator of ω-automata for strategy-controlled models with an iteration like the Kleene star"""

	def __init__(self, module, initial, strategy, aprops, bdd_dict, metamodule=None, opaques=()):
		self.instance = OpSemKleeneInstance.make_instance(module, metamodule)

		super().__init__(self.instance.make_graph(initial, strategy, opaques), aprops, bdd_dict)

		# Table of distinct iterations (from their context to the index of
		# its accepting condition)
		self.iter_map = {}

		self._init_model_checking(module)

	def build(self):
		"""Create an ω-automaton for the rewrite graph"""

		# A transition-based ω-automaton graph (in C++ we could have defined
		# this structure on the fly, but not from Python)
		twa = spot.make_twa_graph(self.bdd_dict)

		# Register and get the BDD variables for the atomic propositions
		self._register_aprops(twa)

		# Depth-first exploration of the graph while creating the TωA
		pending = [0]
		self.state_map = {0: twa.new_state()}

		# self._state_label(self.graph.getStateTerm(0))

		while pending:
			state = pending.pop()
			state_spot = self.state_map[state]

			for next_state in self.graph.getNextStates(state):
				next_state_spot = self.state_map.get(next_state)

				if next_state_spot is None:
					next_state_spot = twa.new_state()
					self.state_map[next_state] = next_state_spot
					pending.append(next_state)

				# Check the iteration tags and set the accepting labels of the edges
				next_state_term = self.instance.get_cterm(self.graph.getStateTerm(next_state))
				acc_set = []

				for tag, enter in self.instance.extract_tags(self.graph.getStateTerm(next_state)):
					acc_index = self.iter_map.get(tag)

					if acc_index is None:
						acc_index = len(self.iter_map)
						self.iter_map[tag] = acc_index

					acc_set.append(2 * acc_index + (0 if enter else 1))

				twa.new_edge(state_spot, next_state_spot,
				             self._state_label(next_state_term),
				             acc_set)

		twa.set_init_state(self.state_map[0])

		# Set the acceptance condition
		acc_condition = ' & '.join(f'(Fin({2 * n}) | Inf({2 * n + 1}))'
		                           for n in range(len(self.iter_map)))

		if acc_condition != '':
			twa.set_acceptance(2 * len(self.iter_map), acc_condition)

		return twa


# Define the translations to the different formulae
# supported by Spot.
spot_dict = {
	'True':		(spot.formula.tt, False),
	'False':	(spot.formula.ff, False),
	'O_':		(spot.formula.X, False),
	'<>_':		(spot.formula.F, False),
	'`[`]_':	(spot.formula.G, False),
	'~_':		(spot.formula.Not, False),
	'_/\\_':	(spot.formula.And, True),
	'_\\/_':	(spot.formula.Or, True),
	'_U_':		(spot.formula.U, False),
	'_R_':		(spot.formula.R, False)
}


def translate_formula(formula):
	"""Translate a formula to Spot"""
	head, *rest = formula

	if head == 'Prop':
		return spot.formula.ap(*rest)
	else:
		ctor, gets_list = spot_dict[head]
		args = [translate_formula(arg) for arg in rest]

		return ctor(args) if gets_list else ctor(*args)


def get_ltl_automaton(formula):
	"""Get the Büchi automaton for the (negation of the) given LTL formula"""

	spot_formula = translate_formula(formula)

	# This translates LTL formulae to Büchi automata
	translator = spot.translator()
	return translator.run(spot.formula.Not(spot_formula))


class SpotBackend:
	"""Spot backend"""

	def find(self):
		return True

	def check(self, graph=None, module=None, metamodule_str=None, term=None, strategy=None, formula=None,
	          logic=None, aprop_terms=None, kleene_iteration=False, get_graph=False, opaque=(), **kwargs):
		"""Run pyModelChecking on the given input problem"""

		# Create the graph if not provided by the caller
		if graph is None:
			graph = create_graph(logic='LTL', term=term, strategy=strategy,
			                     metamodule_str=metamodule_str, opaque=opaque, **kwargs)

		# Reparse the atomic propositions
		if aprop_terms is None:
			aprops = set()
			collect_aprops(formula, aprops)
			aprop_terms = [module.parseTerm(prop) for prop in aprops]

		# Get the Büchi automaton for the formula
		formula_automaton = get_ltl_automaton(formula)

		# Get the Kripke structure for the model
		if kleene_iteration:
			model_builder = KleeneModelBuilder(module, term, strategy, aprop_terms,
			                                   formula_automaton.get_dict(),
			                                   metamodule=metamodule_str,
			                                   opaques=opaque)
		else:
			model_builder = StandardModelBuilder(graph, aprop_terms, formula_automaton.get_dict())

		model_automaton = model_builder.build()

		# Record the time when Spot actually starts model checking
		start_time = time.perf_counter_ns()

		# Model check them, i.e. find whether the intersection is empty
		run = model_automaton.intersecting_run(formula_automaton)

		# It returns either an accepting run or None
		holds = run is None
		stats = {
			'states': model_builder.graph.getNrStates(),
			'rewrites': model_builder.getNrRewrites(),
			'buchi': formula_automaton.num_states(),
			'backend_start_time': start_time
		}

		if graph.strategyControlled:
			stats['real_states'] = model_builder.graph.getNrRealStates()

		if not holds:
			stats['counterexample'] = model_builder.extract_counterexample(run, model_automaton)

		if get_graph:
			stats['graph'] = model_builder.graph if not kleene_iteration \
				else OpSemGraph(model_builder.graph, model_builder.instance)

		return holds, stats
