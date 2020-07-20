#
# Graph generation command
#

import os
import contextlib
import subprocess
import sys
from shutil import which

from ..backend.nusmv import NuSMVGrapher
from ..common import maude, usermsgs, default_model_settings, parse_initial_data, split_comma
from ..wrappers import wrapGraph
from ..formatter import get_formatters, print_term


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


class ProcessStream:
	"""Process wrapper as a context for its input stream"""

	def __init__(self, process):
		self.process = process

	def __enter__(self):
		return self.process.stdin

	def __exit__(self, exc_type, exc_value, traceback):
		self.process.stdin.close()
		self.process.wait()
		return False


def output_stream(filename, extra_args):
	"""Determine the output stream based on the input parameters"""
	if filename is None:
		return contextlib.nullcontext(sys.stdout)
	else:
		if os.path.splitext(filename)[1] == '.pdf':
			dotpath = which('dot')
			if dotpath is None:
				usermsgs.print_warning('GraphViz does not seem to be installed. Writing DOT instead of PDF.')
			else:
				dotp = subprocess.Popen(['dot', '-Tpdf', '-o', filename] + extra_args, stdin=subprocess.PIPE, text=True)
				return ProcessStream(dotp)

		return open(filename, 'w')


def graph(args):
	"""Graph subcommand"""

	initial_data = parse_initial_data(args)

	if initial_data is None:
		return 1

	# Some relevant flags
	withStrategy = args.strategy is not None
	toNuSMV      = args.o is not None and os.path.splitext(args.o)[1] == '.smv'
	purge_fails, merge_states = default_model_settings('CTL' if toNuSMV else 'LTL', args.purge_fails,
							   args.merge_states, args.strategy, tableau=toNuSMV)

	if not withStrategy:
		graph = maude.RewriteGraph(initial_data.term)

	else:
		graph = maude.StrategyRewriteGraph(initial_data.term, initial_data.strategy,
						   initial_data.opaque, initial_data.full_matchrew)

	# It is possible to generate models of the NuSMV model checker instead of DOT graphs
	# They are can be annotated with atomic propositions that are checked in the states.
	if args.aprops is not None:
		aprops = [initial_data.module.parseTerm(prop) for prop in split_comma(args.aprops)]

		if None in aprops:
			usermsgs.print_warning('Some atomic propositions cannot be parsed. Ignoring.')
			aprops.remove(None)
	else:
		aprops = []

	slabel, elabel = get_formatters(args.slabel, args.elabel, withStrategy, only_labels=True)

	with output_stream(args.o, args.extra_args) as outfile:
		if toNuSMV:
			grapher = NuSMVGrapher(outfile, slabel=slabel, elabel=elabel, aprops=aprops)
		else:
			grapher = DOTGrapher(outfile, slabel=slabel, elabel=elabel)

		graph = wrapGraph(graph, purge_fails, merge_states)
		grapher.graph(graph, bound=args.depth)
