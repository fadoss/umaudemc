#
# Utility to test and benchmark collections of examples
#

import os
import copy		# To copy TestCases in some situations
import csv		# To write the measures collected on benchmarks
import json		# To parse test suite specifications
# import psutil		# To measure memory usage
import re		# To filter files from being process
import time		# To measure time

from ..common import maude, usermsgs, default_model_settings
from ..wrappers import wrapGraph
from ..formulae import Parser, collect_aprops
from ..backends import supported_logics, get_backends, backend_for


class TestCase:
	"""Set of input data defining a model-checking problem"""

	def __init__(self, filename, rel_filename):
		self.line_number = None
		self.filename = filename
		self.rel_filename = rel_filename
		self.module_str = None
		self.module = None
		self.metamodule_str = None
		self.metamodule_term = None
		self.metamodule = None
		self.initial_str = None
		self.initial = None
		self.strategy_str = None
		self.strategy = None
		self.opaque = []
		self.formula = None
		self.labels = None
		self.formula_str = None
		self.ftype = None
		self.expected = None

	@property
	def model_module(self):
		return self.module if self.metamodule is None else self.metamodule


def load_cases(filename):
	"""Load test cases from YAML or JSON specifications"""

	extension = os.path.splitext(filename)[1]

	# The YAML package is only loaded if YAML will be used because
	# it is an optional dependency
	if extension in {'.yaml', '.yml'}:
		try:
			import yaml
			from yaml.loader import SafeLoader

		except ImportError:
			usermsgs.print_error(
				'Cannot load cases from YAML file, since the yaml package is not installed.\n'
				'Please convert the YAML to JSON or install it with pip install pyaml.')
			return None

		# The YAML loader is replaced so that entities have its line number
		# associated to print more useful messages. This is not possible with
		# the standard JSON library as easily.

		class SafeLineLoader(SafeLoader):
			def construct_mapping(self, node, deep=False):
				mapping = super(SafeLineLoader, self).construct_mapping(node, deep=deep)
				# Add 1 so line numbering starts at 1
				mapping['__line__'] = node.start_mark.line + 1
				return mapping

		try:
			with open(filename, 'r') as caspec:
				return yaml.load(caspec, Loader=SafeLineLoader)

		except yaml.error.YamlError as ype:
			usermsgs.print_error('Error while parsing tests file: ' + str(ype) + '.')

	# JSON format
	else:
		try:
			with open(filename, 'r') as caspec:
				return json.load(caspec)

		except json.JSONDecodeError as jde:
			usermsgs.print_error('Error while parsing tests file: ' + str(jde) + '.')

	return None


def read_suite(filename):
	"""Generator that yields generators on the test cases for each file"""
	cases = load_cases(filename)

	if cases is None:
		return None, None

	# The formula parser
	parser = Parser()

	if not parser.is_ok():
		return None, None

	# The root directory where tests are located
	root = cases.get('root', '.')

	# The list of cases
	suite = cases.get('suite', None)

	if suite is None:
		usermsgs.print_error('Expected key suite in the root of the test specification.')
		return None, None

	for source in suite:
		rel_filename = source.get('file', None)

		if rel_filename is None:
			usermsgs.print_error('Missing key file in test case.')
			return None, None

		filename = os.path.join(root, source['file'])

		yield rel_filename, read_cases(filename, source, parser)

# The following funtions are similar: they receive a test and a string
# for one of its attributes, which are parsed and set within the test.
# In case of error, a message is shown.

def _parse_metamodule(test, metamodule_str):
	if metamodule_str:
		test.metamodule_str  = metamodule_str
		test.metamodule_term = test.module.parseTerm(metamodule_str)

		if test.metamodule_term is None:
			usermsgs.print_error(f'Cannot parse metamodule term {metamodule_str} '
					     f'in module {str(test.module)} of file {test.rel_filename}.')
			return False

		test.metamodule_term.reduce()
		test.metamodule = test.module.downModule(test.metamodule_term)

		if test.metamodule is None:
			usermsgs.print_error(f'Cannot use metamodule {metamodule_str} in file {test.rel_filename}.')
			return False
	return True


def _parse_initial(test, initial_str):
	if initial_str:
		test.initial_str = initial_str
		test.initial     = test.model_module.parseTerm(initial_str)

		if test.initial is None:
			usermsgs.print_error(f'Cannot parse initial term {initial_str} in file {test.rel_filename}.')
			return False
	return True


def _parse_strategy(test, strategy_str):
	if strategy_str:
		test.strategy_str = strategy_str
		test.strategy     = test.model_module.parseStrategy(strategy_str)

		if test.strategy is None:
			usermsgs.print_error(f'Cannot parse strategy {strategy_str} in file {test.rel_filename}.')
			return False
	return True


def _parse_opaque(test, opaque_str):
	if opaque_str:
		test.opaque    = opaque_str.split(',')
	return True


def _parse_formula(test, formula_str, parser):
	if formula_str:
		test.formula_str = formula_str
		test.formula, test.ftype = parser.parse(formula_str)
		test.labels = parser.labels

		if test.formula is None or test.ftype == 'invalid':
			usermsgs.print_error(f'Bad formula {test.formula_str} in file {test.rel_filename}.')
			return False
	return True


def read_cases(filename, source, parser):
	"""Generator that reads the test cases for each file"""

	source_cases = source.get('cases', None)
	rel_filename = source['file']

	if source_cases is None:
		usermsgs.print_error(f'Expected cases for file {rel_filename}.')
		return None

	if not maude.load(filename):
		usermsgs.print_error(f'Cannot load {rel_filename}.')
		return None

	test = TestCase(filename, rel_filename)

	test.module_str = source.get('module')

	# Get the current module if no module name is specified
	test.module = maude.getModule(test.module_str) if test.module_str else maude.getCurrentModule()

	if test.module is None:
		if test.module_str:
			usermsgs.print_error(f'Cannot find module {test.module_str} in file {rel_filename}.')
		else:
			usermsgs.print_error(f'No module found in file {rel_filename}.')

		return None, None

	# Metamodule
	if not _parse_metamodule(test, source.get('metamodule')):
		return None, None

	parser.set_module(test.model_module,
			  test.metamodule_term)

	if not _parse_initial(test, source.get('initial')) or \
	   not _parse_strategy(test, source.get('strategy')) or \
	   not _parse_formula(test, source.get('formula'), parser) or \
	   not _parse_opaque(test, source.get('opaque')):
		return None, None

	oldftype = 'X'

	for case in source_cases:
		test2 = copy.copy(test)
		test2.line_number = case['__line__']

		module_name = case.get('module')
		metamodule  = case.get('metamodule')
		initial     = case.get('initial')
		strategy    = case.get('strategy')
		formula     = case.get('formula')
		opaque      = case.get('opaque')

		# If the module or metamodule have been overwritten in the case,
		# all other data inherited from the file level must be parsed
		# again in the new module. Setting a module discards the metamodule
		# of the upper level, since this seems more natural.

		parser2 = parser

		if module_name is not None or metamodule is not None:

			if module_name is not None:
				test2.module     = maude.getModule(module_name)
				test2.module_str = module_name

				if test2.module is None:
					usermsgs.print_error(f'Cannot find module {test.module_str} in file {rel_filename}.')
					return None, None

			# Discard the metamodule (if any) and set the current module

			test2.metamodule      = None
			test2.metamodule_str  = None
			test2.metamodule_term = None

			if not _parse_metamodule(test2, metamodule):
				return None, None

			parser2 = copy.copy(parser)

			parser2.set_module(test.module if test.metamodule is None else test.metamodule,
					   test.metamodule_term)

			# Invalidate all that has been calculated with the previous module
			# (and reparse it if not overwritten in the case)

			test2.initial = None
			if test2.initial_str is not None and initial is None and \
			   not _parse_initial(test2, test2.initial_str):
				return None, None

			test2.strategy = None
			if test2.strategy_str is not None and strategy is None and \
			   not _parse_strategy(test2, test2.strategy_str):
				return None, None

			test2.formula = None
			if test2.formula_str is not None and formula is None and \
			   not _parse_formula(test2, test2.formula_str, parser2):
				return None, None

		if not _parse_initial(test2, initial) or \
		   not _parse_strategy(test2, strategy) or \
		   not _parse_formula(test2, formula, parser2) or \
		   not _parse_opaque(test2, opaque):
			return None, None

		if test2.initial is None or test2.formula is None:
			usermsgs.print_error(f'Missing initial state or formula for a test case in file {rel_filename}.')
			return None, None

		test2.expected = case.get('result')

		# Whether the model has changed
		changed_model = {None} != {module_name, metamodule, initial, strategy, opaque} or test2.ftype[0] != oldftype[0]

		yield test2, changed_model


def _run_backend(name, backend, case, args):
	aprops = set()
	collect_aprops(case.formula, aprops)

	return backend.check(module=case.model_module,
			     module_str=case.module_str,
			     metamodule_str=case.metamodule_str,
			     term=case.initial,
			     term_str=case.initial_str,
			     strategy=case.strategy,
			     strategy_str=case.strategy_str,
			     opaque=case.opaque,
			     full_matchrew=False,
			     formula=case.formula,
			     formula_str=case.formula_str,
			     logic=case.ftype,
			     labels=case.labels,
			     filename=case.filename,
			     aprops=aprops,
			     purge_fails=args.purge_fails,
			     merge_states=args.merge_states,
			     get_graph=False,
			     extra_args=args.extra_args)

def run_tests(suite, backends, args, only_file=None, only_logic=None):
	for source, cases in suite:
		if source is None:
			return 1

		# Ignore files that does not match pattern
		if only_file and not only_file.match(source):
			continue

		print(source)
		for case, changed in cases:

			if case is None:
				return 1

			# Ignore the formula if only-logic is present
			if only_logic and not case.ftype.lower() in only_logic:
				continue

			name, backend = backend_for(backends, case.ftype)

			if name is None:
				usermsgs.print_warning(f'No supported backend for {case.ftype} formulae.'
						       f'Skipping {case.formula_str} in {source}.')
				continue

			result, stats = _run_backend(name, backend, case, args)

			if result != case.expected:
				if case.line_number:
					print(f'Line {case.line_number}: ', end='')
				print(f'In {case.initial_str}, {case.module_str}, {case.formula_str}, '
				      f'{case.strategy}: expected {case.expected} but get {result}.')

				if args.fatal_errors:
					return 2

	return 0


def benchmark_tests(suite, backends, args, only_file=None, only_logic=None, out_file='test.csv'):
	with open(out_file, 'w') as logfile:
		log = csv.writer(logfile)

		log.writerow([
			'filename', 'module', 'initial', 'formula', 'strategy',
			'opaque', 'expected', 'logic', 'backend', 'result',
			'states', 'buchi', 'time_ms'
		])

		for source, cases in suite:
			if source is None:
				return 1

			# Ignore files that does not match pattern
			if only_file and not only_file.match(source):
				continue

			print(source)
			for case, changed in cases:

				if case is None:
					return 1

				# Ignore the formula if only-logic is present
				if only_logic and not case.ftype.lower() in only_logic:
					continue

				for name, backend in backends:
					if case.ftype not in supported_logics[name]:
						continue

					# Measure time
					start_time = time.perf_counter_ns()
					result, stats = _run_backend(name, backend, case, args)
					end_time = time.perf_counter_ns()

					if result is not None:
						log.writerow([
							case.rel_filename,
							case.module,
							case.initial_str,
							case.formula_str,
							case.strategy_str,
							' '.join(case.opaque),
							case.expected,
							case.ftype,
							name,
							result,
							stats.get('states'),
							stats.get('buchi'),
							(end_time - start_time) / 1e6
						])


def test(args):
	"""Test subcommand"""
	maude.init(advise=False)

	try:
		# These are filter that limit which case are checked based on filenames or logics
		only_file  = re.compile(args.only_file) if args.only_file else None
		only_logic = args.only_logic.lower().split(',') if args.only_logic else None

		backends, unavailable = get_backends(args.backend)
		if len(unavailable) > 0:
			usermsgs.print_warning('The following backends have not been found: '
				+ ' '.join((name for name, backend in unavailable)))

		if args.benchmark:
			return benchmark_tests(read_suite(args.file), backends, args, only_file, only_logic)
		else:
			return run_tests(read_suite(args.file), backends, args, only_file, only_logic)

	except FileNotFoundError as fnfe:
		usermsgs.print_error('File not found: ' + str(fnfe) + '.')

	except re.error as rerr:
		usermsgs.print_error('Bad regular expression passed to --only-file: ' + str(rerr) + '.')
		return 1

	return 1
