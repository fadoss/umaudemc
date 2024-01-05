#
# Entry point of the umaudemc tool
#
# Its command line arguments are defined here.
#

import argparse
import importlib.util
import sys

from . import usermsgs
from .__init__ import __version__


def add_initial_data_args(parser):
	"""Arguments for the basic input data of a model-checking problem"""

	parser.add_argument(
		'-m', '--module',
		help='specify the module for model checking',
		metavar='NAME'
	)
	parser.add_argument(
		'-M', '--metamodule',
		help='specify a metamodule for model checking',
		metavar='TERM'
	)
	parser.add_argument(
		'--opaque',
		help='opaque strategy names (comma-separated)',
		metavar='LIST',
		default=''
	)
	parser.add_argument(
		'--full-matchrew',
		help='enable full matchrew trace generation',
		action='store_true'
	)
	parser.add_argument(
		'--purge-fails',
		help='remove states where the strategy has failed from the model',
		choices=['default', 'yes', 'no'],
		default='default'
	)
	parser.add_argument(
		'--merge-states',
		help='avoid artificial branching due to strategies by merging states',
		choices=['default', 'state', 'edge', 'no'],
		default='default'
	)


def add_label_format_args(parser):
	"""Arguments to specify the format of labels"""

	parser.add_argument('--slabel', help='state label format specification', metavar='FORMAT')
	parser.add_argument('--elabel', help='edge label format specification', metavar='FORMAT')


def add_backend_arg(parser):
	"""Argument to specify the prioritized list of backends"""

	parser.add_argument(
		'--backend',
		help='comma-separated prioritized list of model-checking backends '
		     '(among maude, ltsmin, pymc, nusmv, spot, spin, builtin)'
	)


def build_parser():
	"""Build the command-line argument parser"""

	parser = argparse.ArgumentParser(description='Maude model checker helper utility')
	subparsers = parser.add_subparsers(help='Program options')

	#
	# Arguments for the whole program
	#

	parser.add_argument(
		'--no-advise',
		help='suppress debug messages from Maude',
		dest='advise',
		action='store_false'
	)
	parser.add_argument(
		'--version',
		help='print version information',
		dest='version',
		action='store_true'
	)
	parser.add_argument(
		'-v', '--verbose',
		help='show additional messages',
		action='store_true'
	)

	parser.set_defaults(mode='gui', web=False, address=None, rootdir=None, sourcedir=None, no_browser=None)

	#
	# Graphical interface
	#

	parser_gui = subparsers.add_parser('gui', help='Graphical interface')

	parser_gui.add_argument(
		'--web',
		help='use web interface even if Gtk is available',
		action='store_true'
	)
	parser_gui.add_argument(
		'--sourcedir',
		help='initial source directory'
	)
	parser_gui.add_argument(
		'--rootdir',
		help='restrict filesystem access to the given directory (implies web)'
	)
	parser_gui.add_argument(
		'--address',
		help='server listening address and port (implies web)',
		metavar='ADDRESS:PORT'
	)
	parser_gui.add_argument(
		'--no-browser',
		help='do not open a browser automatically',
		action='store_true'
	)
	add_backend_arg(parser_gui)

	parser_gui.set_defaults(mode='gui', web=False)

	#
	# Model check from the command line
	#

	parser_check = subparsers.add_parser('check', help='Check a temporal property')

	parser_check.add_argument('file', help='Maude source file specifying the model-checking problem')
	parser_check.add_argument('initial', help='initial term')
	parser_check.add_argument('formula', help='temporal formula')
	parser_check.add_argument('strategy', help='strategy expression', nargs='?')

	add_initial_data_args(parser_check)
	parser_check.add_argument(
		'-k', '--kleene-iteration',
		help='makes the iteration semantics coincide with that of the Kleene star',
		action='store_true'
	)
	parser_check.add_argument(
		'--show-strat',
		help='shows the next strategy to be executed for each state in the counterexample',
		action='store_true'
	)
	add_backend_arg(parser_check)
	parser_check.add_argument(
		'-c', '--counterexample',
		help='reorder the backends to favor those providing counterexamples',
		action='store_true'
	)
	add_label_format_args(parser_check)
	parser_check.add_argument(
		'-f', '--format',
		help='format for printing the counterexample',
		choices=['text', 'json', 'dot', 'html']
	)
	parser_check.add_argument(
		'-o',
		help='output counterexample to a file',
		metavar='FILENAME'
	)

	parser_check.set_defaults(mode='check')

	#
	# Graph generator
	#

	parser_graph = subparsers.add_parser('graph', help='Graph a model')
	parser_graph.add_argument('file', help='Maude source file')
	parser_graph.add_argument('initial', help='initial term')
	parser_graph.add_argument('strategy', help='strategy expression', nargs='?')

	add_initial_data_args(parser_graph)
	parser_graph.add_argument(
		'--format',
		help='select the output format',
		choices=['default', 'dot', 'tikz', 'nusmv', 'prism', 'spin', 'jani'],
		default='default'
	)
	parser_graph.add_argument('-o', help='output to a file', metavar='FILENAME')
	add_label_format_args(parser_graph)
	parser_graph.add_argument(
		'--depth',
		help='depth exploration bound (-1 for unbounded)',
		type=int,
		default=-1
	)
	parser_graph.add_argument(
		'--aprops',
		help='comma-separated list of atomic propositions to be written as annotations if supported',
		metavar='LIST'
	)
	parser_graph.add_argument(
		'--passign',
		help='probability assignment method for adding them to the graph',
		metavar='METHOD'
	)
	parser_graph.add_argument(
		'-k', '--kleene-iteration',
		help='write annotations on the transitions starting and terminating an iteration',
		action='store_true'
	)
	parser_graph.set_defaults(mode='graph')

	#
	# Probabilistic and quantitative model checking
	#

	parser_pcheck = subparsers.add_parser('pcheck', help='Probabilistic and quantitative model checking')

	parser_pcheck.add_argument('file', help='Maude source file specifying the model-checking problem')
	parser_pcheck.add_argument('initial', help='initial term')
	parser_pcheck.add_argument('formula', help='temporal formula')
	parser_pcheck.add_argument('strategy', help='strategy expression', nargs='?')

	add_initial_data_args(parser_pcheck)

	parser_pcheck.add_argument(
		'--assign',
		help='Assign probabilities to the successors according to the given method',
		metavar='METHOD',
		default='uniform'
	)
	parser_pcheck.add_argument(
		'--steps',
		help='Calculate the mean cost in number of steps',
		action='store_true'
	)
	parser_pcheck.add_argument(
		'--reward',
		help='Calculate the expected value of the given reward term on states '
		     '(it may contain a single variable to be replaced by the state term)',
		metavar='TERM'
	)
	parser_pcheck.add_argument(
		'--raw-formula',
		help='The formula argument is directly passed to the backend',
		action='store_true'
	)
	parser_pcheck.add_argument(
		'--backend',
		help='comma-separated prioritized list of probabilistic model-checking backends (among prism, storm)'
	)
	parser_pcheck.add_argument(
		'--fraction', '-f',
		help='show approximated fractional probabilities',
		action='store_true'
	)

	parser_pcheck.set_defaults(mode='pcheck')

	#
	# Statistical model checking
	#

	parser_scheck = subparsers.add_parser('scheck', help='Statistical model checking')

	parser_scheck.add_argument('file', help='Maude source file specifying the model-checking problem')
	parser_scheck.add_argument('initial', help='initial term')
	parser_scheck.add_argument('query', help='QuaTEx query')
	parser_scheck.add_argument('strategy', help='strategy expression', nargs='?')

	add_initial_data_args(parser_scheck)

	parser_scheck.add_argument(
		'--assign',
		help='Assign probabilities to the successors according to the given method',
		metavar='METHOD'
	)
	parser_scheck.add_argument(
		'--alpha', '-a',
		help='Complement of the confidence level (probability outside the confidence interval)',
		type=float,
		default=0.05
	)
	parser_scheck.add_argument(
		'--delta', '-d',
		help='Maximum admissible radius for the confidence interval',
		type=float,
		default=0.5
	)
	parser_scheck.add_argument(
		'--block', '-b',
		help='Number of simulations before checking the confidence interval',
		type=int,
		default=30
	)
	parser_scheck.add_argument(
		'--nsims', '-n',
		help='Number of simulations (it can be a fixed number or a range min-max, where any of the limits can be omitted)',
		default='30-'
	)
	parser_scheck.add_argument(
		'--seed', '-s',
		help='Random seed',
		type=int
	)
	parser_scheck.add_argument(
		'--jobs', '-j',
		help='Number of parallel simulation threads',
		type=int,
		default=1
	)
	parser_scheck.add_argument(
		'--format', '-f',
		help='Output format for the simulation results',
		choices=['text', 'json'],
		default='text'
	)
	parser_scheck.add_argument(
		'--plot', '-p',
		help='Plot the results of parametric queries (using Matplotlib)',
		action='store_true'
	)

	parser_scheck.set_defaults(mode='scheck')

	#
	# Test and benchmark test suites from the command line
	#

	parser_test = subparsers.add_parser('test', help='Test model-checking examples')
	parser_test.add_argument('file', help='Test suite specification (in JSON or YAML format)')

	add_backend_arg(parser_test)
	parser_test.add_argument(
		'--only-file',
		help='Restrict testing to specific files',
		metavar='REGEX'
	)
	parser_test.add_argument(
		'--only-logic',
		help='Restrict testing to formulae in certain logics (comma-separated list)',
		metavar='LIST'
	)
	parser_test.add_argument(
		'--benchmark',
		help='Execute the example with all the given backends and measure it',
		action='store_true'
	)
	parser_test.add_argument(
		'--memory',
		help='Measure memory consumption by each backend (incompatible with benchmark)',
		action='store_true'
	)
	parser_test.add_argument(
		'--fatal-errors',
		help='Stop when the first error is discovered',
		action='store_true'
	)
	parser_test.add_argument(
		'--repeats',
		help='Repeat benchmark tests a number of times (by default 1)',
		default='1'
	)
	parser_test.add_argument(
		'--purge-fails', help='remove states where the strategy has failed from the model',
		choices=['default', 'yes', 'no'],
		default='default'
	)
	parser_test.add_argument(
		'--merge-states',
		help='avoid artificial branching due to strategies by merging states',
		choices=['default', 'state', 'edge', 'no'],
		default='default'
	)
	parser_test.add_argument(
		'-o',
		help='select the name of the CSV file where benchmark results are saved'
	)
	parser_test.add_argument(
		'--from',
		help='start executing the test cases from a given file and case (file:case index)',
		dest='fromcase'
	)
	parser_test.add_argument(
		'--timeout',
		help='timeout for test case executions',
		type=int,
		default=None
	)
	parser_test.add_argument(
		'--no-resume',
		help='do not resume the execution of the test suite after a timeout',
		dest='resume',
		action='store_false'
	)
	parser_test.add_argument(
		'--memory-method',
		help='choose the method for measuring memory usage (they are not equivalent)',
		choices=['memusage', 'psutil'],
		default='memusage'
	)
	parser_test.add_argument(
		'--maudebin',
		help='measure the memory usage of an external Maude binary rather than the builtin maude backend'
	)

	parser_test.set_defaults(mode='test')

	return parser


def main():
	"""umaudemc entry point"""

	# Separate extra arguments to be passed to the external tool (if any)
	if '--' in sys.argv:
		index = sys.argv.index('--')
		own_args = sys.argv[1:index]
		extra_args = sys.argv[index+1:]
	else:
		own_args = sys.argv[1:]
		extra_args = []

	parser = build_parser()
	args = parser.parse_args(args=own_args)
	args.extra_args = extra_args

	if args.version:
		print('Unified Maude model-checking tool ' + __version__)

		dependencies = {'maude', 'pyModelChecking', 'gi', 'yaml', 'spot'}
		available = {dep for dep in dependencies if importlib.util.find_spec(dep) is not None}

		print('\nInstalled dependencies: ' + ' '.join(sorted(available)))
		if available != dependencies:
			print('Missing dependencies: ' + ' '.join(sorted(dependencies - available)))

		return 0

	# Enable colored output (by enabling ANSI escape sequence processing) in Windows
	if sys.platform == 'win32':
		import ctypes
		kernel32 = ctypes.windll.kernel32
		# -11 is standard output, 4 is ENABLE_VIRTUAL_TERMINAL_PROCESSING
		kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 4)

	has_gtk = importlib.util.find_spec('gi')
	has_maude = importlib.util.find_spec('maude')

	if not has_maude:
		usermsgs.print_error('The maude Python package is not available.\n'
		                     'It can be installed with "pip install maude".')
		return 1

	# GUI interface subcommand
	if args.mode == 'gui':

		# Check whether Gtk is really installed if required (the current has_gtk
		# only tells whether the GObject introspection library is installed)

		use_web = args.web or args.address is not None or args.rootdir is not None

		if has_gtk and not use_web:
			try:
				from . import gtk
			except ModuleNotFoundError:
				has_gtk = False
			except Exception as e:
				usermsgs.print_warning(f'Loading web interface instead of Gtk: {e}')
				has_gtk = False

		# If Gtk is not installed or if explicitly requested,
		# the browser-based interface is used

		if has_gtk and not use_web:
			return gtk.run_gtk()
		else:
			from . import webui
			return webui.run(args)

	# Graph generation subcommand
	elif args.mode == 'graph':
		from .command.graph import graph
		return graph(args)

	# Model-checking subcommand
	elif args.mode == 'check':
		from .command.check import check
		return check(args)

	# Probabilistic model-checking subcommand
	elif args.mode == 'pcheck':
		from .command.pcheck import pcheck
		return pcheck(args)

	# Statistical model-checking subcommand
	elif args.mode == 'scheck':
		from .command.scheck import scheck
		return scheck(args)

	# Batch test subcommand
	elif args.mode == 'test':
		from .command.test import test
		return test(args)

	# This should never happen
	else:
		usermsgs.print_error(f'No such subcommand {args.mode}.')

	return 1


if __name__ == '__main__':
	sys.exit(main())
