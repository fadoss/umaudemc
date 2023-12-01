#
# Graph generation command
#

import contextlib
import os
import subprocess
import sys
from shutil import which

from ..common import maude, usermsgs, default_model_settings, parse_initial_data, split_comma
from ..formatter import get_formatters
from ..grapher import DOTGrapher, PDOTGrapher, TikZGrapher
from ..wrappers import wrapGraph


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


def deduce_format(args):
	"""Deduce the output format of the graph"""

	oformat, ofile = args.format, args.o

	# Deduce the output format from the extension of the output file
	extension = os.path.splitext(ofile)[1] if ofile is not None else ''

	if oformat == 'default':
		if extension == '.smv':
			return 'nusmv'
		elif extension in ['.dot', '.pdf']:
			return 'dot'
		elif extension == '.tex':
			return 'tikz'
		elif extension == '.pm':
			return 'prism'
		elif extension == '.pml':
			return 'spin'
		elif extension == '.jani':
			return 'jani'

	elif extension == '.pdf' and oformat != 'dot':
		usermsgs.print_warning('PDF output is only supported with DOT. Changing to DOT.')
		return 'dot'

	# Probabilitic information is not considered in some cases
	if args.passign and oformat not in ('dot', 'prism', 'jani', 'default'):
		usermsgs.print_warning('Probabilistic graphs are only supported with PRISM, JANI and DOT output. '
		                       'Ignoring probabilities.')
		args.passign = None

	return oformat


def graph(args):
	"""Graph subcommand"""

	initial_data = parse_initial_data(args)

	if initial_data is None:
		return 1

	# Some relevant flags
	with_strategy = args.strategy is not None
	oformat = deduce_format(args)
	purge_fails, merge_states = default_model_settings('CTL' if oformat in ('nusmv', 'prism') else 'LTL',
	                                                   args.purge_fails,
	                                                   args.merge_states,
	                                                   args.strategy)

	# Select the appropriate rewriting graph
	# (probabilistic, strategy-controlled, or standard)
	if args.passign or oformat in ('prism', 'jani'):
		from ..probabilistic import get_probabilistic_graph

		rwgraph = get_probabilistic_graph(initial_data, args.passign or 'uniform',
		                                  allow_file=True,
		                                  purge_fails=args.purge_fails,
		                                  merge_states=args.merge_states)

	elif not with_strategy:
		rwgraph = maude.RewriteGraph(initial_data.term)

	else:
		rwgraph = maude.StrategyRewriteGraph(initial_data.term, initial_data.strategy,
		                                     initial_data.opaque, initial_data.full_matchrew)
		rwgraph = wrapGraph(rwgraph, purge_fails, merge_states)

	# If something has failed when creating the graph
	if rwgraph is None:
		return 1

	# It is possible to generate models of the NuSMV and PRISM model checkers instead of
	# DOT graphs. They can be annotated with atomic propositions that are checked in the states.
	if args.aprops is not None:
		aprops = [initial_data.module.parseTerm(prop) for prop in split_comma(args.aprops)]

		if None in aprops:
			usermsgs.print_warning('Some atomic propositions cannot be parsed. Ignoring.')
			aprops.remove(None)
	else:
		aprops = []

	slabel, elabel = get_formatters(args.slabel, args.elabel, with_strategy, only_labels=True)

	# Whether the graph should be seen as a CTMC and probabilities should not be normalized
	is_ctmc = args.passign and args.passign.startswith('ctmc-')

	with output_stream(args.o, args.extra_args) as outfile:
		if oformat == 'prism':
			from ..backend.prism import PRISMGenerator
			grapher = PRISMGenerator(outfile, aprops=aprops, slabel=slabel, ctmc=is_ctmc)
		elif oformat == 'jani':
			from ..jani import JANIGenerator
			grapher = JANIGenerator(outfile, aprops=aprops, slabel=slabel, ctmc=is_ctmc)
		elif oformat == 'nusmv':
			from ..backend.nusmv import NuSMVGrapher
			grapher = NuSMVGrapher(outfile, slabel=slabel, elabel=elabel, aprops=aprops)
		elif oformat == 'spin':
			from ..backend.spin import SpinGrapher
			grapher = SpinGrapher(outfile, slabel=slabel, elabel=elabel)
		elif oformat == 'tikz':
			grapher = TikZGrapher(outfile, slabel=slabel, elabel=elabel)
		elif args.passign:
			grapher = PDOTGrapher(outfile, slabel=slabel, elabel=elabel, ctmc=is_ctmc)
		else:
			grapher = DOTGrapher(outfile, slabel=slabel, elabel=elabel)

		grapher.graph(rwgraph, bound=args.depth)

	return 0
