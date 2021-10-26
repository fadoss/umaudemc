#
# pyModelChecking backend for umaudemc
#

import time

import pyModelChecking

from .. import usermsgs
from ..formulae import collect_aprops
from ..wrappers import create_graph


class KripkeBuilder:
	"""Builds a Kripke structure for the pyModelChecking library"""

	def __init__(self, graph, aprops):
		"""
		Create a KripkeBuilder.
		:param graph: Rewrite graph
		:type graph: Maude rewrite graphs or their wrapped versions
		:param aprops: Terms for each atomic proposition to be listed in the module
		:type aprops: collection of maude.Term
		"""
		self.graph = graph
		self.aprops = aprops
		module = graph.getStateTerm(0).symbol().getModule()

		self.relation = []
		self.labeling = {}
		self.visited = set()

		# Find the satisfaction (|=) symbol and the true constant to be used
		# when testing atomic propositions.

		bool_kind = module.findSort('Bool').kind()
		state_kind = module.findSort('State').kind()
		prop_kind = module.findSort('Prop').kind()

		self.true = module.parseTerm('true', bool_kind)
		self.satisfies = module.findSymbol('_|=_', (state_kind, prop_kind), bool_kind)

		# Number of rewrites to test them
		self.nrRewrites = 0

	def getNrRewrites(self):
		"""Total count of rewrites used to generate the Kripke structure"""
		return self.nrRewrites + self.graph.getNrRewrites()

	def labels_of(self, state):
		"""Set of atomic propositions holding in a state (Kripke's labeling function)"""

		term = self.graph.getStateTerm(state)
		labels = set()

		for prop in self.aprops:
			t = self.satisfies.makeTerm((term, prop))
			self.nrRewrites += t.reduce()
			if t == self.true:
				labels.add(str(prop))

		return labels

	def expand(self, initial_state):
		"""Explore the graph reachable from state filling the Kripke structure"""
		pending = [initial_state]
		self.visited.add(initial_state)

		while pending:
			state = pending.pop()

			self.labeling[state] = self.labels_of(state)
			deadlock = True

			for next_state in self.graph.getNextStates(state):
				deadlock = False
				self.relation.append([state, next_state])

				# Depth-first search with a stack
				if next_state not in self.visited:
					self.visited.add(next_state)
					pending.append(next_state)

			# The transition relation must be total, so self-loops are added
			# when there is no successors. Notice that failed states should
			# be purged when using pyModelChecking even for LTL.
			if deadlock:
				self.relation.append([state, state])

	def make_kripke(self):
		"""Get the Kripke structure for the given input problem"""
		self.relation = []
		self.labeling = {}
		self.visited = set()

		self.expand(0)

		return pyModelChecking.kripke.Kripke(S=list(self.visited),
		                                     S0={0},
		                                     R=self.relation,
		                                     L=self.labeling)

# Define the translations to the different formulae
# supported by pyModelChecking.

import pyModelChecking.LTL.language as ltl_lang
import pyModelChecking.CTL.language as ctl_lang
import pyModelChecking.CTLS.language as ctls_lang

ltl_translation = {
	'Prop'		: ltl_lang.AtomicProposition,
	'True'		: lambda: ltl_lang.Bool(True),
	'False'		: lambda: ltl_lang.Bool(False),
	'~_'		: ltl_lang.Not,
	'`[`]_'		: ltl_lang.G,
	'<>_'		: ltl_lang.F,
	'O_'		: ltl_lang.X,
	'_/\\_'		: ltl_lang.And,
	'_\\/_'		: ltl_lang.Or,
	'_<->_'		: lambda f, g: ltl_lang.And(ltl_lang.Imply(f, g), ltl_lang.Imply(g, f)),
	'_->_'		: ltl_lang.Imply,
	'_U_'		: ltl_lang.U,
	'_R_'		: ltl_lang.R
}


def ctl_translation(lang):
	"""Translation for both CTL and CTL*"""

	# The same constructors are used for both but in a
	# different namespace.

	return {
		'Prop'		: lang.AtomicProposition,
		'True'		: lambda: lang.Bool(True),
		'False'		: lambda: lang.Bool(False),
		'~_'		: lang.Not,
		'E_'		: lang.E,
		'A_'		: lang.A,
		'`[`]_'		: lang.G,
		'<>_'		: lang.F,
		'O_'		: lang.X,
		'_/\\_'		: lang.And,
		'_\\/_'		: lang.Or,
		'_<->_'		: lambda f, g: lang.And(lang.Imply(f, g), lang.Imply(g, f)),
		'_->_'		: lang.Imply,
		'_U_'		: lang.U,
		'_R_'		: lang.R
	}


def make_formula(formula, translation):
	"""Translate a formula from a translation"""

	head, *rest = formula
	if head == 'Prop':
		return translation['Prop'](*rest)
	else:
		return translation[head](*[make_formula(f, translation) for f in rest])


class PyModelChecking:
	"""pyModelChecking backend"""

	def find(self):
		return True

	def check_kripke(self, kripke, formula, ftype):
		"""Check a given temporal property"""

		if ftype in ['LTL', 'propLogic']:
			from pyModelChecking.LTL.model_checking import modelcheck
			pymc_formula = ltl_lang.A(make_formula(formula, ltl_translation))
		elif ftype == 'CTL':
			from pyModelChecking.CTLS.model_checking import modelcheck
			pymc_formula = make_formula(formula, ctl_translation(ctl_lang))
		elif ftype == 'CTL*':
			from pyModelChecking.CTLS.model_checking import modelcheck
			pymc_formula = make_formula(formula, ctl_translation(ctls_lang))
		else:
			raise ValueError('Unexpected logic for pyModelChecking backend.')

		# pyModelChecking supports fairness constraints, but they are not used

		try:
			# Record the time when pyModelChecking actually starts
			start_time = time.perf_counter_ns()
			sset = modelcheck(kripke, pymc_formula)

		except RuntimeError as re:
			usermsgs.print_error('An error has occurred when running pyModelChecking:\n' + str(re))
			return None, None

		# The result of pyModelChecking's modelcheck function is the set of states
		# in which the property holds. This is added to the statistics.

		stats = {'sset': sset, 'backend_start_time': start_time}

		return (0 in sset), stats

	def check(self, graph=None, module=None, formula=None, logic=None, aprop_terms=None, get_graph=False, **kwargs):
		"""Run pyModelChecking on the given input problem"""

		# Create the graph if not provided by the caller
		if graph is None:
			graph = create_graph(logic=logic, tableau=True, **kwargs)

		# Reparse the atomic propositions
		# (atomic propositions terms that were parsed as part of the formula
		# cannot be directly used because there were parsed in a different module)
		if aprop_terms is None:
			aprops = set()
			collect_aprops(formula, aprops)

			aprop_terms = [module.parseTerm(prop) for prop in aprops]

		kbuilder = KripkeBuilder(graph, aprop_terms)
		kripke = kbuilder.make_kripke()
		holds, stats = self.check_kripke(kripke, formula, logic)

		if holds is not None:
			stats['states'] = graph.getNrStates()
			if graph.strategyControlled:
				stats['real_states'] = graph.getNrRealStates()
			stats['rewrites'] = kbuilder.getNrRewrites()
		if get_graph:
			stats['graph'] = graph

		return holds, stats
