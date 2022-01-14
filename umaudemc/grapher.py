#
# Class to generate graphs in the GraphViz's DOT format
#

import sys
from fractions import Fraction

from .formatter import print_term


class DOTGrapher:
	"""Graph writer in GraphViz's DOT format"""

	def __init__(self, outfile=sys.stdout, slabel=print_term, elabel=None):
		self.visited = set()
		self.outfile = outfile
		self.slabel = slabel
		self.elabel = elabel

	def write_transition(self, start, end, elabel):
		"""Write a transition from start to end"""

		if elabel:
			elabel = str(elabel).replace('"', '\\"')
			print(f'\t{start} -> {end} [label="{elabel}"];', file=self.outfile)
		else:
			print(f'\t{start} -> {end};', file=self.outfile)

	def write_state(self, graph, state):
		"""Write the label of a state"""

		print(f'\t{state} [label="{self.slabel(graph, state)}"];', file=self.outfile)

	def graph(self, graph, bound=-1):
		print('digraph {', file=self.outfile)
		self.exploreAndGraph(graph, 0, bound)
		print('}', file=self.outfile)

	def exploreAndGraph(self, graph, stateNr, bound=-1):
		self.visited.add(stateNr)
		self.write_state(graph, stateNr)

		if bound == 0:
			return

		for next_state in graph.getNextStates(stateNr):
			elabel = self.elabel(graph, stateNr, next_state) if self.elabel else None
			self.write_transition(stateNr, next_state, elabel)

			if next_state not in self.visited:
				self.exploreAndGraph(graph, next_state, -1 if bound == -1 else bound-1)


class PDOTGrapher(DOTGrapher):
	"""Graph writer in GraphViz's DOT format for probabilistic models"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.visited = {}

	def make_label(self, p, graph, start, end):
		"""Build an extended label with the probability and standard label"""

		# Probabilities are printed as fractions for legibility
		num, den = Fraction(p).limit_denominator().as_integer_ratio()

		# Standard labels are used too
		elabel = self.elabel(graph, start, end)

		return f'{num}/{den} {elabel}' if den > 1 else elabel

	def exploreAndGraph(self, graph, stateNr, bound=-1):
		# Visited is a dictionary from state number to depth
		self.visited[stateNr] = 0
		self.write_state(graph, stateNr)

		# StrategyMarkovGraph does not record transition labels,
		# so we make a dirty fix
		if not hasattr(graph, 'getTransition'):
			class FakeTransition:
				def getType(self):
					return 100

			graph.getTransition = lambda *_: FakeTransition()

		for state, children in graph.transitions():
			depth = self.visited.get(state)

			if depth is None or 0 <= bound <= depth:
				continue

			for p, child in children:
				child_depth = self.visited.get(child)

				# New state or lower depth, set its depth
				if child_depth is None or depth + 1 < child_depth:
					self.visited[child] = depth + 1

				# New state, print its label
				if child_depth is None:
					self.write_state(graph, child)

				elabel = self.make_label(p, graph, state, child)
				self.write_transition(state, child, elabel)


class TikZGrapher:
	"""Graph writer in TikZ format"""

	def __init__(self, outfile=sys.stdout, slabel=print_term, elabel=None):
		self.visited = set()
		self.outfile = outfile
		self.slabel = slabel
		self.elabel = elabel

	def graph(self, graph, bound=-1):
		print('''% Requires the tikz package and its graphs and quotes libraries
\\begin{tikzpicture}[solution/.style={}]
\\graph {''', file=self.outfile)
		self.exploreAndGraph(graph, 0, bound)
		print('};\n\\end{tikzpicture}', file=self.outfile)

	def exploreAndGraph(self, graph, stateNr, bound=-1):
		if bound == 0:
			return

		for next_state in graph.getNextStates(stateNr):
			print(f'\t{self.printState(graph, stateNr)}', file=self.outfile, end='')
			next_visited = next_state in self.visited

			if self.elabel is None:
				print(f' -> ', file=self.outfile, end='')
			else:
				label = str(self.elabel(graph, stateNr, next_state)).replace('"', '""')
				print(f' ->["{label}"] ', file=self.outfile, end='')

			print(f'{self.printState(graph, next_state)};', file=self.outfile)

			if not next_visited:
				self.exploreAndGraph(graph, next_state, -1 if bound == -1 else bound-1)

	def printState(self, graph, stateNr):
		name = f's{stateNr}'

		if stateNr not in self.visited:
			label = str(self.slabel(graph, stateNr)).replace('"', '""')
			name += f'/"{label}"'
			self.visited.add(stateNr)

		return name
