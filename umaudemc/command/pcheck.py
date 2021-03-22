#
# Command-line probabilistic model-checking subcommand
#

from ..backend import prism
from ..backends import format_statistics
from ..common import parse_initial_data, usermsgs
from ..formulae import ProbParser
from ..probabilistic import get_assigner, RewardEvaluator
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


def _is_ctl(formula):
	"""Check whether a formula is not a linear-time one"""
	head, *args = formula

	if head == 'Prop':
		return False

	elif head in ('A_', 'E_'):
		return True

	else:
		start = 1 if head in ('P__', '<>__', '`[`]__', '_U__', '_W__', '_R__') else 0
		return any((_is_ctl(arg) for arg in args[start:]))


def pcheck(args):
	"""Probabilistic check subcommand"""

	# Parse the model-checking problem data
	data = parse_initial_data(args)

	if data is None:
		return 1

	# Parse the formula if not a raw one
	if not args.raw_formula:
		formula, aprops = _parse_formula(data, args)

		if formula is None:
			return 1

		# CTL and PCTL formula must get their states merged in the
		# strategy-controlled case
		ftype = 'CTL' if _is_ctl(formula) else 'LTL'
	else:
		formula, aprops, ftype = None, None, 'raw'

	# Initialize the PRISM backend
	backend = prism.PRISMBackend()

	if not backend.find():
		usermsgs.print_error('PRISM cannot be found (after searching in the system path and the PRISM_PATH variable).\n'
		                     'It can be downloaded from https://www.prismmodelchecker.org/.')

		return 1

	# Get the probability assignment method
	distr, found = get_assigner(data.module, args.assign)

	if distr is None:
		if not found:
			usermsgs.print_error(f'Unknown probability assignment method {args.dist}.')
		return 1

	# Get the optional reward evaluation function by parsing the reward term
	reward = None

	if args.reward is not None:
		reward_term = data.module.parseTerm(args.reward)

		if reward_term is None:
			usermsgs.print_warning('The reward term cannot be parsed. It will be ignored.')

		else:
			reward = RewardEvaluator(reward_term)

	# Solve the given problem
	result, stats = backend.check(module=data.module,
	                              metamodule=data.metamodule,
	                              term=data.term,
	                              formula=formula,
	                              formula_str=args.formula,
	                              strategy=data.strategy,
	                              logic=ftype,
	                              aprops=aprops,
	                              purge_fails=args.purge_fails,
	                              merge_states=args.merge_states,
	                              extra_args=args.extra_args,
	                              cost=args.steps,
	                              reward=reward,
	                              dist=distr)

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
					print(f'Result: {vmin} to {vmax}{_print_extra(result)}')
					return 0

			print(f'Result: {result.value}{_print_extra(result)}')

	return 0
