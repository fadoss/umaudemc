#
# Utility to test and benchmark collections of examples
#

import os
import copy       # To copy TestCases in some situations
import csv        # To write the measures collected on benchmarks
import itertools  # To iterate in the product of the parameter values
import json		  # To parse test suite specifications
# import psutil	  # To measure memory usage
import re         # To filter files from being process
import time       # To measure time

from ..common import maude, usermsgs
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
		self.result = None

		self.exclude = []
		self.parameter = None
		self.parser = None

	@property
	def model_module(self):
		return self.module if self.metamodule is None else self.metamodule

	@property
	def expected(self):
		return self.result


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

		except yaml.error.YAMLError as ype:
			usermsgs.print_error(f'Error while parsing test file: {ype}.')

	# JSON format
	else:
		try:
			with open(filename, 'r') as caspec:
				return json.load(caspec)

		except json.JSONDecodeError as jde:
			usermsgs.print_error(f'Error while parsing test file: {jde}.')

	return None


def read_suite(filename):
	"""Generator that reads a test specification file and yields
	   generators on the test cases for each file"""

	# The YAML or JSON dictionary with the test specification structure
	cases = load_cases(filename)

	if cases is None:
		return None, None

	# The parser for temporal formulae
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

	elif not isinstance(suite, list):
		usermsgs.print_error('The suite key in the test specification must be a list.')
		return None, None

	for source in suite:
		# The name of the file (relative to the previous root)
		rel_filename = source.get('file', None)

		if rel_filename is None:
			usermsgs.print_error('Missing key file in test case.')
			return None, None

		filename = os.path.join(root, source['file'])

		yield rel_filename, read_cases(filename, source, parser)

# The following functions are similar: they receive a test and a string
# for one of its attributes, which is parsed and set within the test.
# In case of error, a message is shown.


def _store_value(test, name, value, handler):
	setattr(test, (name + '_str') if handler is not None else name, value)


def _parse_module(test):
	if test.module_str:
		test.module = maude.getModule(test.module_str)

		if test.module is None:
			usermsgs.print_error(f'Cannot find module {test.module_str} in file {test.rel_filename}.')

	else:
		test.module = maude.getCurrentModule()

		if test.module is None:
			usermsgs.print_error(f'No module found in file {test.rel_filename}.')

	return test.module is not None


def _parse_metamodule(test):
	test.metamodule_term = test.module.parseTerm(test.metamodule_str)

	if test.metamodule_term is None:
		usermsgs.print_error(f'Cannot parse metamodule term {test.metamodule_str} '
		                     f'in module {str(test.module)} of file {test.rel_filename}.')
		return False

	test.metamodule_term.reduce()
	test.metamodule = maude.downModule(test.metamodule_term)

	if test.metamodule is None:
		usermsgs.print_error(f'Cannot use metamodule {test.metamodule_str} in file {test.rel_filename}.')
		return False

	return True


def _parse_initial(test):
	if test.initial_str is None:
		return True

	test.initial = test.model_module.parseTerm(test.initial_str)

	if test.initial is None:
		usermsgs.print_error(f'Cannot parse initial term {test.initial_str} in file {test.rel_filename}.')
		return False

	return True


def _parse_strategy(test):
	if test.strategy_str is None:
		return True

	test.strategy = test.model_module.parseStrategy(test.strategy_str)

	if test.strategy is None:
		usermsgs.print_error(f'Cannot parse strategy {test.strategy_str} in file {test.rel_filename}.')
		return False

	return True


def _parse_formula(test):
	if test.formula_str is None:
		return True

	test.formula, test.ftype = test.parser.parse(test.formula_str)
	test.labels = test.parser.labels

	if test.formula is None or test.ftype == 'invalid':
		usermsgs.print_error(f'Bad formula {test.formula_str} in file {test.rel_filename}.')
		return False

	return True


def _filter_backends(backends, excluded):
	"""Remove backends included in the excluded collection"""

	if excluded:
		return [(name, backend) for name, backend in backends if name not in excluded]
	else:
		return backends


def _depends_on_vars(node, variables):
	"""Whether the given variables occur at the given node"""

	for value in node.values():
		if isinstance(value, str) and any([f'${var}' in value for var in variables]):
			return True
	return False


class ParameterSet:
	"""Assignement to parameters to be replaced on test attributes"""

	def __init__(self, dic, is_subs=None):
		# Dictionary with the parameter assignments
		self.dic = dic
		# Remove the __line__ mark if present (YAML)
		if '__line__' in dic:
			dic.pop('__line__')
		# Whether this is a substitution
		self.is_substitution = self._is_substitution() if is_subs is None else is_subs

	def _is_substitution(self):
		"""Decides whether this a substitution or includes alternatives values for each variable"""

		return all((not isinstance(val, list) for val in self.dic.values()))

	@classmethod
	def combine(cls, parent, child):
		"""Combine two parameter sets"""

		if child is None:
			return parent
		elif parent is None:
			return cls(child)
		else:
			return cls({**parent.dic, **child})

	def apply(self, text):
		"""Replace the parameters in a given string"""

		# Only substitutions can be properly applied
		if not self.is_substitution:
			return text

		for var, value in self.dic.items():
			if f'${var}' in text:
				text = text.replace(f'${var}', str(value))

		return text

	def instances(self):
		"""Generate the substitutions for all possible combinations of the parameter values"""

		return map(lambda vls: ParameterSet(dict(zip(self.dic.keys(), vls)), is_subs=True),
		           itertools.product(*self.dic.values()))

	@property
	def variables(self):
		"""Variables included in this parameter set"""

		return set(self.dic.keys())

	def __repr__(self):
		return repr(self.dic)


# Fields admitted by the test specification format
# (some others keys are admitted but treated apart)
#
# They consists of a name, a type, the function that need to be called for
# parsing them, and the arguments that must be reparsed because of it.

TEST_FIELDS = {
	'module':	(str, _parse_module, {'update-parser', 'initial', 'strategy', 'formula'}),
	'metamodule':	(str, _parse_metamodule, {'update-parser', 'initial', 'strategy', 'formula'}),
	'initial':	(str, _parse_initial, set()),
	'strategy':	(str, _parse_strategy, set()),
	'formula':	(str, _parse_formula, set()),
	'opaque':	(list, None, {'update-parser'}),
	'result':	(bool, None, set()),
	'exclude':	(list, None, set())
}

# Order to parse case elements
# (the last three can be permuted at choice)
PARSING_ORDER = ('module', 'metamodule', 'update-parser', 'initial', 'strategy', 'formula')


def read_cases(filename, source, parser):
	"""Generator that reads the test cases for each file"""

	# A stack is used to explore the tree of cases. It consists of the
	# parent test object, an iterator to the case subtree, the current
	# case subtree and an iterator to the pending substitutions.
	test_stack = [(None, iter([source]), None, itertools.repeat(None))]

	# Type of the last formula
	oldftype = 'X'

	while test_stack:
		ptest, iterator, source, subs = test_stack.pop()

		# Copy the parent test object (because it may overwrite
		# some attributes)
		test = copy.copy(ptest)

		# Get the next substitution to try (if any)
		substitution = next(subs, None)

		# If there is no (more) substitutions, try to
		# get the next test case (or arrangement)
		if substitution is None:
			source = next(iterator, None)

		# If there is no (more) test cases, continue down the stack
		if source is None:
			continue

		# Set of fields that will need to be parsed at this level
		touched = set()

		# Filenames are only allowed in the first level
		if not test_stack:
			rel_filename = source['file']

			if not maude.load(filename):
				usermsgs.print_error(f'Cannot load {rel_filename}.')
				return None, None

			test = TestCase(filename, rel_filename)
			test.parser = parser
			touched.update({'module', 'update-parser'})

		elif 'file' in source:
			usermsgs.print_warning('Filenames not allowed in nested case specifications. Ignoring.')

		# Get the test case line (not currently available for JSON)
		test.line_number = source.get('__line__', 0)

		# Handle parameters
		# (when substitution is None, because otherwise they have already been handled)

		if substitution is None:
			# Get the parameter annotation at the current level
			parameter = source.get('parameter')

			if parameter is not None and not isinstance(parameter, dict):
				usermsgs.print_error('The parameter value must be a dictionary.')
				return None, None

			# Combine the parameters bindings of the parent node if any
			parameter = ParameterSet.combine(test.parameter, parameter)

			if parameter is not None:
				# Set the parameter attribute of the test
				test.parameter = parameter

				# If the parameter is a substitution, it will be applied,
				# but otherwise we have to generate the substitutions by
				# combination if they are actually used at this level

				if not parameter.is_substitution and \
				   _depends_on_vars(source, parameter.variables):
					subs = parameter.instances()
					substitution = next(subs, None)

					if substitution is None:
						usermsgs.print_error('Parameter defined with a empty range.')
						return None, None

					test.parameter = substitution

		else:
			test.parameter = substitution

		#
		# Process fields
		#

		for key, value in source.items():
			# These are handled elsewhere
			if key in {'file', '__line__', 'cases', 'parameter'}:
				continue

			# Get the type and instruction of what we have to do with this field
			etype, parse_fn, deps = TEST_FIELDS.get(key, (None,) * 3)

			if etype is None:
				usermsgs.print_warning(f'Ignoring unknown key {key} in case at line {source["__line__"]}.')
				continue

			# Rudimentary type checking

			if etype is list and not isinstance(value, list):
				# Single elements are upgraded to lists
				value = [value]

			elif etype is str and isinstance(value, int) or isinstance(value, float):
				# Something atomic and convertible to a string is
				# admitted as string, but dictionaries and lists are not
				value = str(value)

			elif key == 'strategy' and value is None:
				# Allowed to indicate that no strategy is used
				pass

			elif not isinstance(value, etype):
				usermsgs.print_error(f'Unexpected value type for key {key} (expected {etype.__name__} '
				                     f'but found {type(value).__name__}).')
				return None, None

			# Instantiate the raw values with the substitution (if any)
			if test.parameter is not None and isinstance(value, str):
				value = test.parameter.apply(value)

			# Store the raw test case information
			_store_value(test, key, value, parse_fn)

			# Mark key and its dependencies as touched
			touched.update(deps)
			touched.add(key)

		# Invalidates the metamodule if a module was given but not a metamodule,
		# so that it gets reparsed. Setting a module discards the metamodule
		# of the upper level, since this seems more natural.
		if 'module' in source and 'metamodule' not in source:
			test.metamodule_str = None
			test.metamodule = None

		# Update those fields in the test that are either new or need
		# to be reparsed due changes in other fields on which they
		# depend (essentially the module).

		for action in filter(lambda a: a in touched, PARSING_ORDER):
			# Update the temporal formulae parser
			if action == 'update-parser':
				test.parser = copy.copy(test.parser)
				test.parser.set_module(test.model_module,
				                       test.metamodule_term)

			# Otherwise, a field of the problem is to be parsed
			else:
				parse_fn = TEST_FIELDS.get(action)[1]

				if not parse_fn(test):
					return None, None

		# The case may be a leaf node, which must provide all required
		# information for the model checking problem, or it may contain
		# nested cases, which inherit their fields from this
		source_cases = source.get('cases', None)

		if source_cases is None:
			# Check whether all required information is available
			if test.initial is None or test.formula is None:
				usermsgs.print_error((f'Missing initial state or formula for a test case in file {test.rel_filename}'
				                      f'(line {test.line_number}).') if len(test_stack) > 0 else
				                      f'Expected cases for file {test.rel_filename}.')
				return None, None

			# Whether the model has changed
			changed_model = not touched.isdisjoint({'module', 'metamodule', 'initial', 'strategy', 'opaque'}) \
			                or test.ftype[0] != oldftype[0]

			test_stack.append((ptest, iterator, source, subs))
			yield test, changed_model

		else:
			# Keep exploring the siblings of the current node later
			test_stack.append((ptest, iterator, source, subs))
			# First explore the children of the current node
			test_stack.append((test, iter(source_cases), None, itertools.repeat(None)))


def _run_backend(name, backend, case, args):
	"""Run the test case in the given model-checking backend"""

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
	"""Run the tests in the suite with the first backend available"""

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

			# Filter backends with the excluded attribute
			filtered_backends = _filter_backends(backends, case.exclude)

			name, backend = backend_for(filtered_backends, case.ftype)

			if name is None:
				usermsgs.print_warning(f'No supported backend for {case.ftype} formulae. '
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


def benchmark_tests(suite, backends, args, only_file=None, only_logic=None, out_file='test.csv', repeats=1):
	"""Benchmark the model checkers with the test suite"""

	# A CSV file is written with a row for each test case and backend
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

				# Filter backends with the excluded attribute
				filtered_backends = _filter_backends(backends, case.exclude)

				for name, backend in filtered_backends:
					if case.ftype not in supported_logics[name]:
						continue

					print('  ', case.formula_str, 'on', case.initial_str, 'with', name)

					for rtp in range(repeats):

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
	"""Entry point for the test subcommand"""

	maude.init(advise=args.advise)

	try:
		# These are filter that limit which case are checked based on filenames or logics
		only_file = re.compile(args.only_file) if args.only_file else None
		only_logic = args.only_logic.lower().split(',') if args.only_logic else None

		backends, unavailable = get_backends(args.backend)
		if len(unavailable) > 0:
			usermsgs.print_warning('The following backends have not been found: '
			                       + ' '.join((name for name, backend in unavailable)))

		if args.benchmark:
			return benchmark_tests(read_suite(args.file), backends, args, only_file, only_logic, repeats=args.repeats)
		else:
			return run_tests(read_suite(args.file), backends, args, only_file, only_logic)

	except FileNotFoundError as fnfe:
		usermsgs.print_error('File not found: ' + str(fnfe) + '.')

	except re.error as rerr:
		usermsgs.print_error('Bad regular expression passed to --only-file: ' + str(rerr) + '.')
		return 1

	return 1
