#
# Command-line model-checking subcommand
#


import os.path
import sys

from ..common import default_model_settings, parse_initial_data, usermsgs, maude
from ..backends import supported_logics, get_backends, backend_for, LTSmin, format_statistics
from ..counterprint import SimplePrinter, JSONPrinter, HTMLPrinter, DOTPrinter, print_counterexample
from ..wrappers import wrapGraph
from ..formulae import Parser, collect_aprops
from ..formatter import get_formatters


# Emphasized text to be used when printing the model-checking result
_itis  = '\033[1;32mis\033[0m'
_isnot = '\033[1;91mis not\033[0m'


def get_printer(args):
	"""Get printer and formatter according to the program arguments"""

	# Get the state and edge label formatters
	formatters = get_formatters(args.slabel, args.elabel, args.strategy is not None)

	# Obey the format argument is specified
	oformat = args.format

	# Otherwise take the extension of the output file as reference
	if oformat is None and args.o is not None:
		oformat = {
			'.txt'	: 'text',
			'.htm'	: 'html',
			'.html'	: 'html',
			'.json'	: 'json',
			'.dot'	: 'dot'
		}[os.path.splitext(args.o)[1]]

	elif oformat is None:
		oformat = 'text'

	# Opens the output file
	try:
		ofile = open(args.o, 'w') if args.o is not None else sys.stdout

	except OSError as fnfe:
		usermsgs.print_warning(f'Output file cannot be written: {str(fnfe)}. '
			               'Writing to the terminal.')
		ofile = sys.stdout

	if oformat == 'text':
		pass
	elif oformat == 'html':
		return HTMLPrinter(ofile), *formatters
	elif oformat == 'json':
		return JSONPrinter(ofile), *formatters
	elif oformat == 'dot':
		return DOTPrinter(ofile), *formatters
	else:
		usermsgs.print_warning('Counterexample formatter not available. Defaulting to text mode.')

	return SimplePrinter(args.show_strat), *formatters


def logic_name(logic):
	"""User-friendly name of the logic"""
	return 'μ-calculus' if logic == 'Mucalc' else logic


def generic_check(data, args, formula, logic, backend, handle):
	"""Check a model-checking problem using the given backend"""

	# Calculates the atomic propositions in the formula
	aprops = set()
	collect_aprops(formula, aprops)

	# Print the input data if the version option is set
	if args.verbose:
		msg = ['Model checking', logic_name(logic), 'property', args.formula, 'from', data.term] + \
		      (['using', data.strategy] if data.strategy is not None else []) + \
		      ['in module', data.module]

		print(*msg)

	# Call the selected backend
	# (many arguments to satisfy the requirements of all of them)
	holds, stats = handle.check(module=data.module,
				    module_str=args.module,
				    metamodule_str=args.metamodule,
				    term=data.term,
				    term_str=args.initial,
				    strategy=data.strategy,
				    strategy_str=args.strategy,
				    opaque=data.opaque,
				    full_matchrew=data.full_matchrew,
				    formula=formula,
				    formula_str=args.formula,
				    logic=logic,
				    labels=data.labels,
				    filename=args.file,
				    aprops=aprops,
				    purge_fails=args.purge_fails,
				    merge_states=args.merge_states,
				    get_graph=True,
				    extra_args=args.extra_args)

	if holds is None:
		return 4

	print(f'The property {_itis if holds else _isnot} satisfied in the initial state '
	      f'({format_statistics(stats)})')

	if 'counterexample' in stats:
		print_counterexample(stats['graph'], stats['counterexample'], get_printer(args))

	return 0


def suggest_install(backends, ftype):
	"""Suggest installing the first backend that supports the given formula"""

	backend, handler = backend_for(backends, ftype)

	if backend == 'pymc':
		usermsgs.print_error(
			'pyModelChecking cannot be found.\n'
			'It can be installed with pip install pyModelChecking.')
	elif backend == 'ltsmin':
		if not handler.find_ltsmin():
			usermsgs.print_error(
				'LTSmin cannot be found (after searching in the system path and the LTSMIN_PATH variable).\n'
				'It can be downloaded from https://ltsmin.utwente.nl, but in order to use μ-calculus formulae \n'
				'with both edge and state labels, a modified version is required that can be downloaded from\n'
				'http://maude.ucm.estrategies/#downloads.')

		if not handler.find_maudemc():
			usermsgs.print_error(
				f'The Maude plugin for LTSmin cannot be found (libmaudemc{LTSmin.module_suffix}).\n'
				'Setting the environment variable MAUDEMC_PATH to its location helps.\n'
				'It can be downloaded from http://maude.ucm.es/strategies/#downloads.')

	elif backend == 'nusmv':
		usermsgs.print_error(
			'NuSMV cannot be found (after searching in the system path and the NUSMV_PATH variable).\n'
			'It can be downloaded from http://nusmv.fbk.eu.')


def check(args):
	"""Check subcommand"""

	# Parse the model-checking problem data
	data = parse_initial_data(args)

	if data is None:
		return 1

	# Creates a parser for the formula
	parser = Parser()

	if not parser.is_ok():
		return 1

	# The module is not checked for being compatible with model checking
	parser.set_module(data.module, data.metamodule)
	data.labels = parser.labels

	formula, ftype = parser.parse(args.formula)

	if formula is None:
		return 2

	# Get the first available backend for the given formula
	backends, unavailable = get_backends(args.backend)
	name, handle = backend_for(backends, ftype)

	if name is not None:
		return generic_check(data, args, formula, ftype, name, handle)

	usermsgs.print_error('No compatible backend for the given formula.')
	suggest_install(unavailable, ftype)

	return 3
