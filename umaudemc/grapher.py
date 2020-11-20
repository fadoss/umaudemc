#
# Class to generate graphs in the GraphViz's DOT format
#

import sys
from .formatter import print_term


class DOTGrapher:
	"""Graph writer in GraphViz's DOT format"""

	def __init__(self, outfile=sys.stdout, slabel=print_term, elabel=None):
		self.visited = set()
		self.outfile = outfile
		self.slabel = slabel
		self.elabel = elabel

	def graph(self, graph, bound=-1):
		print('digraph {', file=self.outfile)
		self.exploreAndGraph(graph, 0, bound)
		print('}', file=self.outfile)

	def exploreAndGraph(self, graph, stateNr, bound=-1):
		self.visited.add(stateNr)
		print(f'\t{stateNr} [label="{self.slabel(graph, stateNr)}"];', file=self.outfile)

		if bound == 0:
			return

		for next_state in graph.getNextStates(stateNr):
			if self.elabel is None:
				print(f'\t{stateNr} -> {next_state};', file=self.outfile)
			else:
				print(f'\t{stateNr} -> {next_state} [label="{self.elabel(graph, stateNr, next_state)}"];', file=self.outfile)

			if next_state not in self.visited:
				self.exploreAndGraph(graph, next_state, -1 if bound == -1 else bound-1)
