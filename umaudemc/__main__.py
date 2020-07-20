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
	"""Argument to specify the priorized list of backends"""

	parser.add_argument(
		'--backend',
		help='comma-separated priorized list of model-checking backends (among maude, ltsmin, pymc, nusmv, builtin)',
		default='maude,ltsmin,pymc,nusmv,builtin'
	)


parser     = argparse.ArgumentParser(description='Maude model checker helper utility')
subparsers = parser.add_subparsers(help='Program options')

#
# Arguments for the whole program
#

parser.add_argument(
	'--no-advise',
	help='supress debug messages (internal use)',
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
	help='use the web interface (by default)',
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
	'--show-strat',
	help='shows the next strategy to be executed for each state in the counterexample',
	action='store_true'
)
add_backend_arg(parser_check)
add_label_format_args(parser_check)
parser_check.add_argument(
	'-f', '--format',
	help='format for printing the counterxample',
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
parser_graph.set_defaults(mode='graph')

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
	'--fatal-errors',
	help='Stop when the first error is discovered',
	action='store_true'
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

parser_test.set_defaults(mode='test')

#
# Separate extra arguments to be passed to the external tool (if any)
#

if '--' in sys.argv:
	index = sys.argv.index('--')
	own_args = sys.argv[1:index]
	extra_args = sys.argv[index+1:]
else:
	own_args = sys.argv[1:]
	extra_args = []

#
# umaudemc entry point
#

args = parser.parse_args(args=own_args)
args.extra_args = extra_args

if args.version:
	print('Unified Maude model-checking tool ' + __version__)

	dependencies = {'maude', 'pyModelChecking', 'yaml'}
	available     = {dep for dep in dependencies if importlib.util.find_spec(dep) is not None}

	print('\nInstalled dependencies: ' + ' '.join(available))
	if available != dependencies:
		print('Missing dependencies: ' + ' '.join(dependencies - available))

	sys.exit(0)

# Enable colored output (by enable ANSI escape sequence processing) in Windows
if sys.platform == 'win32':
	import ctypes
	kernel32 = ctypes.windll.kernel32
	# -11 is standard output, 4 is ENABLE_VIRTUAL_TERMINAL_PROCESSING
	kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 4)

has_maude = importlib.util.find_spec('maude')

if not has_maude:
	usermsgs.print_error('The maude Python package is not available.\n'
			     'It can be installed with "pip install maude".')
	sys.exit(1)

# GUI interface subcommand
if args.mode == 'gui':
	from . import webui
	webui.run(args)

# Graph generation subcommand
elif args.mode == 'graph':
	from .command.graph import graph
	graph(args)

# Model-checking subcommand
elif args.mode == 'check':
	from .command.check import check
	check(args)

# Batch test subcommand
elif args.mode == 'test':
	from .command.test import test
	test(args)

# This should never happen
else:
	usermsgs.print_error(f'No such subcommand {args.mode}.')
