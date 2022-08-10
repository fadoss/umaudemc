#
# JANI output backend for umaudemc
#

import sys

# Fixed content of the JANI file
JANI_HEADER = '''{
	"jani-version": 1,
	"name": "generated.jani",
	"system": {
		"elements": [ { "automaton": "maude" } ]
	},
	"variables": [{
		"name": "s",
		"initial-value": 0,
		"type": {
			"base": "int",
			"kind": "bounded",
			"lower-bound": 0,
'''


class JANIGenerator:
	"""Generator of JANI models"""

	def __init__(self, outfile=sys.stdout, aprops=(), slabel=None, ctmc=False):
		self.visited = set()
		self.aprops = aprops
		self.outfile = outfile

		self.satisfies = None
		self.true_term = None

		self.slabel = slabel
		self.ctmc = ctmc

		# Number of rewrites used to check atomic propositions
		self.nrRewrites = 0

	def getNrRewrites(self):
		return self.nrRewrites

	def init_aprop(self, graph):
		"""Find the resources required for testing atomic propositions"""

		module = graph.getStateTerm(0).symbol().getModule()
		boolkind = module.findSort('Bool').kind()

		self.satisfies = module.findSymbol('_|=_', (module.findSort('State').kind(),
		                                            module.findSort('Prop').kind()),
		                                   boolkind)

		self.true_term = module.findSymbol('true', (), boolkind).makeTerm(())

	def check_aprop(self, graph, propNr, stateNr):
		"""Check whether a given atomic proposition holds in a state"""

		t = self.satisfies.makeTerm((graph.getStateTerm(stateNr), self.aprops[propNr]))
		self.nrRewrites += t.reduce()
		return t == self.true_term

	def graph(self, graph, bound=None):
		"""Generate a JANI input file"""

		# Find the satisfaction (|=) symbol and the true constant to be used
		# when testing atomic propositions.

		if self.aprops:
			self.init_aprop(graph)

		# Build the model specification in the JANI format
		model_type = 'mdp' if graph.nondeterminism else ('ctmc' if self.ctmc else 'dtmc')
		normalize = model_type == 'dtmc'

		print(JANI_HEADER, end='', file=self.outfile)
		print(f'\t\t\t"upper-bound": {len(graph)}\n\t\t}}\n\t}}],\n\t"type": "{model_type}",', file=self.outfile)

		print('\t"automata": [{"name": "maude", "locations": [{"name": "l"}], '
		      '"initial-locations": ["l"], "edges": [', file=self.outfile)

		first_edge = True
		keyword = 'rate' if self.ctmc else 'probability'

		# For each reachable state, we write a dictionary with all the transitions from it
		for state, children in graph.transitions():
			# Avoid printing a state without transitions
			if not children:
				continue

			if not first_edge:
				self.outfile.write(',')
			else:
				first_edge = False

			# Adds labels to the states for graph generation
			comment = f' "comment": "{self.slabel(graph, state)}",' if self.slabel else ''

			print('\t\t{"location": "l", "guard": {"exp": {"op": "=", "left": "s", '
			      f'"right": {state}}}}},{comment} "destinations": [', file=self.outfile)

			first_dest = True

			# Probabilities are normalized for DTMC
			if normalize:
				children = tuple(children)
				total_w = sum(w for w, _ in children)
				children = ((w / total_w, nexts) for w, nexts in children)

			for p, nexts in children:
				if not first_dest:
					self.outfile.write(',')
				else:
					first_dest = False

				print('\t\t\t{"location": "l", "assignments": [{"ref": "s", "value": '
				      f'{nexts}}}], "{keyword}": {{"exp": {p}}}}}', file=self.outfile)

			print('\t\t]}', file=self.outfile)

		print(']}], "properties": [', file=self.outfile)

		# Define a property for each atomic proposition
		first_prop = True

		for propNr, prop in enumerate(self.aprops):
			if not first_prop:
				self.outfile.write(',')
			else:
				first_prop = False

			# Construct a Boolean expression enumerating the states where the property hold
			expr = "false"

			for state in graph.states():
				if self.check_aprop(graph, propNr, state):
					state_eq = f'{{"op": "=", "left": "s", "right": {state}}}'

					if expr != "false":
						expr = f'{{"op": "∨", "left": {state_eq}, "right": {expr}}}'
					else:
						expr = state_eq

			print(f'{{"name": "{prop}", "expression": {expr}}}', file=self.outfile)

		print(']}', file=self.outfile)
