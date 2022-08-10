#
# Command-line probabilistic model-checking subcommand
#

import fractions
import os
import re

from ..backend import prism, storm
from ..backends import format_statistics
from ..common import parse_initial_data, usermsgs
from ..formulae import ProbParser
from ..probabilistic import get_probabilistic_graph, RewardEvaluator
from ..terminal import terminal as tmn

# Emphasized text to be used when printing Boolean model-checking results
_itis = f'{tmn.bold}{tmn.green}is{tmn.reset}'
_isnot = f'{tmn.bold}{tmn.bright_red}is not{tmn.reset}'


def _parse_formula(data, args):
	"""Parse a probabilistic formula expressed in Maude's language"""
	pparser = ProbParser()

	if not pparser.is_ok():
		return None, None

	# The module is not checked for being compatible with model checking
	pparser.set_module(data.module, data.metamodule)

	return pparser.parse(args.formula)


def _print_extra(result):
	"""Print the additional information of the model-checking result"""

	# If the result is a range, only the information for the first value is shown
	if result.rtype == result.QR_RANGE:
		extra, _ = result.extra
	else:
		extra = result.extra

	return f' (relative error {extra["rel_error"]})' if extra else ''


def _print_fraction(value):
	"""Print a fraction that approximates the given value"""

	num, den = fractions.Fraction(value).limit_denominator().as_integer_ratio()

	return f'{num}/{den}' if num > 0 and den != 1 else num

def _select_backend(known_backends, backend_list):
	"""Get the first available backend according to the user preferences"""

	# If the given backend_list is empty, we read it from the following
	# environment variable, or otherwise we use the defaults
	if backend_list is None:
		backend_list = os.getenv('UMAUDEMC_PBACKEND')

		if backend_list is None:
			backend_list = 'prism,storm'

	# Get the first available backend in the list
	# (first_known is used to show the error message with the first missing option)
	options, first_known, first_available = backend_list.split(','), None, None

	for name in options:
		_, backend, _ = known_backends.get(name, (None,) * 3)

		if backend is None:
			usermsgs.print_warning(f'Unsupported backend "{name}". Ignoring it.')

		elif backend.find():
			first_available = name
			break

		elif first_known is None:
			first_known = name


	if first_available is None:
		if first_known is None:
			usermsgs.print_error('The backend list does not contain any valid option.')

		else:
			be_name, _, be_url = known_backends[first_known]

			usermsgs.print_error(f'{be_name} cannot be found (after searching in the system path and the '
					     f'{be_name.upper()}_PATH variable).\nIt can be downloaded from {be_url}.')

		return None

	return known_backends[first_available][1]


def _do_state_analysis(data, args, backend, graph, step_number, show_p):
	"""Calculate steady-state or transient probabilities"""

	# State probabilities are only computed for DTMCs
	if graph.nondeterminism:
		usermsgs.print_error('Transient and steady-state probabilities are only computed for DTMCs.')
		return 1

	plist, stats = backend.state_analysis(
		  step=step_number,
		  extra_args=args.extra_args,
		  graph=graph
	)

	if plist is None:
		return 1

	# In the strategy-controlled case, we have to merge state
	# probabilities to obtain term probabilities
	if graph.strategyControlled:
		state_dict = {}

		for k, p in enumerate(plist):
			if p > 0:
				term = graph.getStateTerm(k)
				prev = state_dict.get(term, 0.0)
				state_dict[term] = prev + p

	else:
		state_dict = {graph.getStateTerm(k): p for k, p in enumerate(plist) if p > 0}

	# Print the terms with positive probability (without header)
	for term, p in sorted(state_dict.items(), key=lambda item: - item[1]):
		print(f' {show_p(p):<20} {term}')

	return 0


def pcheck(args):
	"""Probabilistic check subcommand"""

	# Parse the model-checking problem data
	data = parse_initial_data(args)

	if data is None:
		return 1

	# Whether standard state analysis is requested
	state_analysis, step_number = False, None

	# Parse the formula if not a raw one
	match = re.fullmatch(r'@(steady|transient)(?:\((\d+)\))?', args.formula)

	# Pseudoformulas for stead-state and transient analysis
	if match:
		name, step = match.groups()
		state_analysis = True

		if name == 'steady':
			if step:
				usermsgs.print_warning(f'The @steady pseudoformula does not take any argument. It will be ignored.')

		elif name == 'transient':
			if step is None:
				usermsgs.print_error('A step number must be given as @transient(step) for transient analysis.')
				return 1

			step_number = int(step)

	# Formulas expressed in the Maude syntax
	elif not args.raw_formula:
		formula, aprops = _parse_formula(data, args)

		if formula is None:
			return 1

		# We do not check the syntactic validity of the formula,
		# the backends will complain
		ftype = 'CTL'

	# Raw formulas in the syntax of the backend
	else:
		formula, aprops, ftype = None, None, 'raw'

	# Available probabilistic model-checking backends
	backends = {
		'prism': ('PRISM', prism.PRISMBackend(), 'https://www.prismmodelchecker.org/'),
		'storm': ('Storm', storm.StormBackend(), 'https://www.stormchecker.org/'),
	}

	backend = _select_backend(backends, args.backend)

	if backend is None:
		return 1

	# Get the probabilistic graph
	graph = get_probabilistic_graph(data, args.assign, allow_file=True,
					purge_fails=args.purge_fails,
					merge_states=args.merge_states)

	if graph is None:
		return 1

	# Whether to show approximate fractions or floating-point numbers
	show_p = _print_fraction if args.fraction else lambda x: x

	# Do state analysis, if required
	if state_analysis:
		return _do_state_analysis(data, args, backend, graph, step_number, show_p)

	# Get the optional reward evaluation function by parsing the reward term
	reward = None

	if args.reward is not None:
		reward_term = data.module.parseTerm(args.reward)

		if reward_term is None:
			usermsgs.print_warning('The reward term cannot be parsed. It will be ignored.')

		else:
			reward = RewardEvaluator.new(reward_term, data.term.getSort().kind())

	# Solve the given problem
	result, stats = backend.check(module=data.module,
	                              formula=formula,
	                              formula_str=args.formula,
	                              logic=ftype,
	                              aprops=aprops,
	                              extra_args=args.extra_args,
	                              cost=args.steps,
	                              reward=reward,
	                              graph=graph,
	                              ctmc=args.assign.startswith('ctmc-'))

	# Show the results and additional information
	if result is not None:
		if result.rtype == result.QR_BOOLEAN:
			print(f'The property {_itis if result.value else _isnot} satisfied ({format_statistics(stats)}).')

		else:
			# Show statistics when in verbose mode
			if args.verbose:
				print(f'Used {format_statistics(stats)}.')

			if result.rtype == result.QR_RANGE:
				vmin, vmax = result.value

				if vmin != vmax:
					print(f'Result: {show_p(vmin)} to {show_p(vmax)}{_print_extra(result)}')
					return 0
				else:
					result.value = vmin

			print(f'Result: {show_p(result.value)}{_print_extra(result)}')

	return 0
