#
# Maude (strategy) model-checking utility program
#
# Common operations required by most commands
#

import os

import maude

from . import usermsgs

#
# Warn about old versions of the maude package installed
#

if hasattr(maude, 'StateTransitionGraph'):
	usermsgs.print_warning('Version 0.3 of the maude package contains bugs related to model checking.\n'
	                       'Please update.')

	# Anyhow, allow using it at their own risk, for which some adaptations are needed
	maude.RewriteGraph = maude.StateTransitionGraph
	maude.StrategyRewriteGraph = maude.StrategyTransitionGraph

	maude.Symbol.__call__ = lambda self, *args: self.makeTerm(args)

if not hasattr(maude, 'Hook'):
	usermsgs.print_warning('Version 0.5 of the maude package adds some useful features for this program.\n'
	                       'Please update.')

	maude.RewriteGraph.getNrRewrites = lambda: 0
	maude.StrategyRewriteGraph.getNrRewrites = lambda: 0
	maude.ModelCheckResult.nrBuchiStates = 0
	maude.downModule = lambda term: term.symbol().getModule().downModule(term)
	maude.Sort.__le__ = lambda self, other: self.leq(other)
	maude.Term.__eq__ = lambda self, other: self.equal(other)

if not hasattr(maude.StrategyRewriteGraph, 'getNrRealStates'):
	maude.StrategyRewriteGraph.getNrRealStates = maude.StrategyRewriteGraph.getNrStates

if not hasattr(maude.Term, 'getVarName'):
	usermsgs.print_warning('Version 1.0 of the maude package adds some useful features for this program.\n'
	                       'Please update.')

	maude.Term.getVarName = lambda self: str(self).split(':')[0] if self.symbol() == self.symbol().getModule().parseTerm(f'$$$:{self.getSort()}').symbol() else None
	maude.Term.isVariable = lambda self: self.getVarName() is None


class InitialData:
	"""Initial data of the model-checking problem"""

	def __init__(self):
		self.module = None
		self.filename = None
		self.term = None
		self.strategy = None
		self.opaque = []
		self.full_matchrew = False
		self.metamodule = None


def find_maude_file_abs(filename):
	"""Find a Maude file with possibly missing extension"""
	for ext in ['', '.maude', '.fm']:
		if os.path.isfile(filename + ext):
			return filename + ext
	return None


def find_maude_file(filename):
	"""Find a Maude file taking MAUDE_LIB into account"""
	if os.path.isabs(filename):
		return find_maude_file_abs(filename)

	# Maude also considers the current working directory
	# and the directory of the Maude binary
	paths = [os.getcwd(), os.path.dirname(maude.__file__)]

	maudelib = os.getenv('MAUDE_LIB')

	if maudelib is not None:
		paths += maudelib.split(os.pathsep)

	for path in paths:
		abspath = os.path.join(path, filename)
		fullname = find_maude_file_abs(abspath)
		if fullname is not None:
			return fullname

	return None


def parse_initial_data(args):
	"""Inits Maude and parse common initial data of a model-checking problem"""
	maude.init(advise=args.advise)

	data = InitialData()

	# Checks whether the file exists
	data.filename = find_maude_file(args.file)

	if data.filename is None:
		usermsgs.print_error('No such file.')
		return None

	if not maude.load(args.file):
		usermsgs.print_error('Error loading file')
		return None

	# Loads the module

	if args.module is None:
		data.module = maude.getCurrentModule()

		if data.module is None:
			usermsgs.print_error('No last module.')
			return None

	else:
		data.module = maude.getModule(args.module)

		if data.module is None:
			usermsgs.print_error(f'Module {args.module} does not exist.')
			return None

	# Loads a metamodule (if required)

	if args.metamodule is not None:
		mt = data.module.parseTerm(args.metamodule)

		if mt is None:
			usermsgs.print_error('Bad parse for metamodule term.')
			return None

		data.metamodule = mt
		data.module = maude.downModule(mt)

		if data.module is None:
			usermsgs.print_error('Bad metamodule.')
			return None

	# Parse the initial term

	data.term = data.module.parseTerm(args.initial)

	if data.term is None:
		usermsgs.print_error('Bad parse for initial term')
		return None

	# Parse the strategy

	if args.strategy is not None:
		data.strategy = data.module.parseStrategy(args.strategy)

		if data.strategy is None:
			usermsgs.print_error('Bad parse for strategy')
			return None

	else:
		data.strategy = None

	# Opaque strategies and full matchrew

	data.opaque = [] if args.opaque == '' else args.opaque.split(',')
	data.full_matchrew = args.full_matchrew

	return data


def split_comma(string):
	"""Split a string as comma-separated list ignoring commas inside parentheses"""
	result = []
	left, right, depth = 0, 0, 0

	while right < len(string):
		if string[right] == '(':
			depth += 1
		elif string[right] == ')':
			depth -= 1
		elif string[right] == ',' and depth == 0:
			result.append(string[left:right])
			left = right + 1

		right = right + 1

	result.append(string[left:])
	return result


def default_model_settings(logic, purge_fails, merge_states, strategy, tableau=False):
	"""Fill the purge-fails and merge-states defaults"""
	purge_fails, merge_states = purge_fails, merge_states

	# The branching-time adaptations does not make sense
	# for strategy-free models
	if strategy is None:
		if merge_states not in {'default', 'no'}:
			usermsgs.print_warning('Merging states does not make sense without a strategy. '
			                       'Ignoring --merge-states value.')
		if purge_fails not in {'default', 'no'}:
			usermsgs.print_warning('Failed states do no make sense without a strategy. '
			                       'Ignoring --purge-fails flag.')

		return 'no', 'no'

	# The tableau variable indicates whether a tableau-based method will
	# be used for LTL model-checking. In that case, failed states have to
	# be purged as for the branching-time logics.

	if purge_fails == 'default':
		purge_fails = 'yes' if logic in {'CTL', 'CTL*', 'Mucalc'} or tableau else 'no'

	if merge_states == 'default':
		merge_states = 'state' if logic.startswith('CTL') else ('edge' if logic == 'Mucalc' else 'no')

	return purge_fails, merge_states
