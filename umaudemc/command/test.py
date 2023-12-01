#
# Utility to test and benchmark collections of examples
#

import os
import copy        # To copy TestCases in some situations
import csv         # To write the measures collected on benchmarks
import itertools   # To iterate in the product of the parameter values
import json        # To parse test suite specifications
import re          # To filter files from being process
import shutil      # To find memusage
import subprocess  # To execute external programs for measuring their memory usage
import sys         # To find the Python executable path
import threading   # To support timeouts
import time        # To measure time

from ..common import maude, usermsgs
from ..formulae import Parser, collect_aprops
from ..backends import supported_logics, get_backends, backend_for
from ..terminal import terminal as tmn

# Regular expression to extract memory data from glibc's memusage
MEMUSAGE_REGEX = re.compile(br'heap total: (\d+), heap peak: (\d+), stack peak: (\d+)')


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

	# The YAML package is only loaded when needed
	# (pyaml is an optional dependency)
	if extension in ('.yaml', '.yml'):
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
		# the standard JSON library.

		class SafeLineLoader(SafeLoader):
			def construct_mapping(self, node, deep=False):
				mapping = super(SafeLineLoader, self).construct_mapping(node, deep=deep)
				# Add 1 so line numbering starts at 1
				mapping['__line__'] = node.start_mark.line + 1
				return mapping

		try:
			with open(filename) as caspec:
				return yaml.load(caspec, Loader=SafeLineLoader)

		except yaml.error.YAMLError as ype:
			usermsgs.print_error(f'Error while parsing test file: {ype}.')

	# TOML format
	if extension == '.toml':
		try:
			import tomllib

		except ImportError:
			usermsgs.print_error(
				'Cannot load cases from TOML file, '
				'which is only available since Python 3.10.')
			return None

		try:
			with open(filename) as caspec:
				return tomllib.load(caspec)

		except tomllib.TOMLDecodeError as tde:
			usermsgs.print_error(f'Error while parsing test file: {tde}.')

	# JSON format
	else:
		try:
			with open(filename) as caspec:
				return json.load(caspec)

		except json.JSONDecodeError as jde:
			usermsgs.print_error(f'Error while parsing test file: {jde}.')

	return None


def read_suite(filename, from_file=None, skip_case=0):
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
		# Cases can be disabled with the disabled field
		if source.get('disabled', False):
			continue

		# The name of the file (relative to the previous root)
		rel_filename = source.get('file', None)

		# Skip entries until the from case is reached
		if from_file is not None:
			if rel_filename != from_file:
				continue
			else:
				from_file = None

		if rel_filename is None:
			usermsgs.print_error('Missing key file in test case.')
			return None, None

		filename = os.path.join(root, source['file'])

		yield rel_filename, read_cases(filename, source, parser, skip=skip_case)

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

	test.formula, test.ftype = test.parser.parse(test.formula_str, opaques=test.opaque)
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
		if isinstance(value, str) and any(f'${var}' in value for var in variables):
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

		return all(not isinstance(val, list) for val in self.dic.values())

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
# They consist of a name, a type, the function that need to be called for
# parsing them, and the arguments that must be reparsed because of it.

TEST_FIELDS = {
	'module':	(str, _parse_module, {'update-parser', 'initial', 'strategy', 'formula'}),
	'metamodule':	(str, _parse_metamodule, {'update-parser', 'initial', 'strategy', 'formula'}),
	'initial':	(str, _parse_initial, set()),
	'strategy':	(str, _parse_strategy, set()),
	'formula':	(str, _parse_formula, set()),
	'opaque':	(list, None, {'update-parser'}),
	'result':	(bool, None, set()),
	'exclude':	(list, None, set()),
	'preload':      (list, None, {'update-parser', 'initial', 'strategy', 'formula'}),
}

# Order to parse case elements
# (the last three can be permuted at choice)
PARSING_ORDER = ('module', 'metamodule', 'update-parser', 'initial', 'strategy', 'formula')


def read_cases(filename, source, parser, skip=0):
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

			# Load auxiliary files first
			aux_files = source.get('preload', [])
			if isinstance(aux_files, str):
				aux_files = [aux_files]

			base_dir = os.path.dirname(filename)

			for aux_filename in aux_files:
				aux_fullname = os.path.join(base_dir, aux_filename)

				if not maude.load(aux_fullname if os.path.exists(aux_fullname) else aux_filename):
					usermsgs.print_error(f'Cannot load auxiliary file {aux_filename}.')
					return None, None

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

			if skip > 0:
				skip -= 1
			else:
				yield test, changed_model

		else:
			# Keep exploring the siblings of the current node later
			test_stack.append((ptest, iterator, source, subs))
			# First explore the children of the current node
			test_stack.append((test, iter(source_cases), None, itertools.repeat(None)))


def _run_backend(name, backend, case, timeout, args):
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
	                     extra_args=args.extra_args,
	                     timeout=timeout)


def run_tests(suite, backends, args, only_file=None, only_logic=None, resume=None, **_):
	"""Run the tests in the suite with the first backend available"""

	for source, cases in suite:
		if source is None:
			return 1

		# Ignore files that do not match pattern
		if only_file and not only_file.fullmatch(source):
			continue

		# Show the file that is being tested
		print(f'{tmn.bold}{source}{tmn.reset}')

		# Get track of the last executed case
		resume.filename = source

		for case, changed in cases:

			# If there was an error in the test specification
			if case is None:
				return 1

			resume.case_number += 1

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

			resume.start()
			result, stats = _run_backend(name, backend, case, resume.timeout, args)
			resume.stop()

			if result != case.expected:
				print(f'{tmn.bright_red}Wrong result', end='')
				# The line number of the test may not be available in JSON
				if case.line_number:
					print(f' in line {case.line_number}', end='')
				print(f':{tmn.reset} expected {case.expected} but got {result} for')

				# Print the meaningful attributes of the test case
				print(f'  initial term: {case.initial_str}\n', end='')
				if case.module_str is not None:
					print(f'  module: {case.module_str}\n', end='')
				print(f'  formula: {case.formula_str}\n', end='')
				if case.strategy_str is not None:
					print(f'  strategy: {case.strategy_str}\n', end='')
				if case.opaque:
					print(f'  opaque: {",".join(case.opaque)}\n', end='')

				if args.fatal_errors:
					return 2

		resume.case_number = 0

	return 0


def _benchmark_case(case, name, backend, timeout, args, log):
	"""Benchmark a single test case"""

	# Measure time
	start_time = time.perf_counter_ns()
	result, stats = _run_backend(name, backend, case, timeout, args)
	end_time = time.perf_counter_ns()

	if result is not None:
		# Calculate preparation time (if available)
		if stats is not None and 'backend_start_time' in stats:
			backend_start_time = stats.get('backend_start_time', None)
			preparation_time = (backend_start_time - start_time) / 1e6
		else:
			preparation_time = None

		# Calculate the length of the counterexample (if any)
		if stats is not None and 'counterexample' in stats:
			lead_in, cycle = stats['counterexample']
			counter_length = len(lead_in) + len(cycle)
		else:
			counter_length = None

		log.writerow((
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
			stats.get('real_states'),
			stats.get('buchi'),
			counter_length,
			stats.get('rewrites'),
			(end_time - start_time) / 1e6,
			preparation_time
		))

	return (end_time - start_time) / 1e6


def benchmark_tests(suite, backends, args, only_file=None, only_logic=None,
                    out_file='test.csv', repeats=None, resume=None):
	"""Benchmark the model checkers with the test suite"""

	# If continuing a previous execution, we append to the previous log
	open_mode = 'a' if resume.filename is not None and os.path.exists(out_file) else 'w'

	# A CSV file is written with a row for each test case and backend
	with open(out_file, open_mode) as logfile:
		log = csv.writer(logfile)

		# Header of that CSV
		if open_mode != 'a':
			log.writerow((
				'filename', 'module', 'initial', 'formula', 'strategy',
				'opaque', 'expected', 'logic', 'backend', 'result',
				'states', 'real_states', 'buchi', 'clength', 'rewrites', 'time_ms', 'prep_ms'
			))

		for source, cases in suite:
			if source is None:
				return 1

			# Ignore files that do not match pattern
			if only_file and not only_file.fullmatch(source):
				continue

			print(f'{tmn.bold}{source}{tmn.reset}')

			# Get track of the last executed case
			resume.filename = source

			for case, changed in cases:

				if case is None:
					return 1

				resume.case_number += 1

				# Ignore the formula if only-logic is present
				if only_logic and not case.ftype.lower() in only_logic:
					continue

				# Filter backends with the excluded attribute
				filtered_backends = _filter_backends(backends, case.exclude)

				for name, backend in filtered_backends:
					if case.ftype not in supported_logics[name]:
						continue

					# Print some information to see what is going on
					if case.strategy_str is not None:
						strategy_info = f' {tmn.cyan}@{tmn.reset} {case.strategy}'
						if case.opaque:
							strategy_info += ' (opaque)'
					else:
						strategy_info = ''

					print(f'  {case.formula_str} {tmn.cyan}on{tmn.reset} '
					      f'{case.initial_str}{strategy_info} '
					      f'{tmn.cyan}with{tmn.reset} {name}')

					resume.start()
					run_time = _benchmark_case(case, name, backend, resume.timeout, args, log)
					resume.stop()

					# Executes as many repeats as given by the repeats function
					repetitions = repeats(run_time) if repeats is not None else 0

					for rtp in range(1, repetitions):
						resume.start()
						_benchmark_case(case, name, backend, resume.timeout, args, log)
						resume.stop()

			resume.case_number = 0
	return 0


def _print_size(size):
	"""Print memory size in multiples of a byte"""

	if size < 1e3:
		return str(size)

	elif size < 1e6:
		return '{:.2f} Kb'.format(size / 1e3)

	elif size < 1e9:
		return '{:.2f} Mb'.format(size / 1e6)

	else:
		return '{:.2f} Gb'.format(size / 1e9)


def _make_self_call(case, name):
	"""Prepare a self call to umaudemc for a given test case"""

	# If we are running umaudemc from the zipped executable file, this
	# file should probably be included in the PYTHONPATH for this to work
	self_call = [sys.executable, '-m', 'umaudemc', 'check', '--backend', name,
	             case.filename, case.initial_str, case.formula_str]

	if case.strategy_str is not None:
		self_call.append(case.strategy_str)

	if case.module_str is not None:
		self_call.append(f'--module={case.module_str}')

	if case.metamodule_str is not None:
		self_call.append(f'--metamodule={case.metamodule_str}')

	if case.opaque:
		self_call.append(f'--opaque=' + ','.join(case.opaque))

	return self_call


def _feed_maude_binary(p, case):
	"""Feed the Maude binary with the test case"""

	# Represent the opaque names as a QidList term (as a string)
	opaques = ' '.join("'" + name for name in case.opaque) if case.opaque else 'nil'

	# Issue a select command if the module is not the default one
	if case.module_str is not None:
		p.stdin.write(f'select {case.module_str} .\n'.encode('utf-8'))

	# Use the standard model checker term if no strategy was given
	# (MODEL-CHECK must be included in the module)
	if case.strategy_str is None:
		# If the module is an object-level module, we simply reduce the modelCheck symbol
		if case.metamodule is None:
			p.stdin.write(f"red modelCheck({case.initial_str}, {case.formula_str}) .\n".encode('utf-8'))

		# If the module in the test case is a metamodule, we should reduce it at the metalevel
		else:
			metalevel = maude.getModule('META-LEVEL')
			formula_term = case.metamodule.parseTerm(case.formula_str)
			p.stdin.write(f'red in META-LEVEL : metaReduce({case.metamodule_str}, '
				f'modelCheck[{metalevel.upTerm(case.initial)}, {metalevel.upTerm(formula_term)}]) .\n'.encode('utf-8'))

	# Use the strategy-aware model checker term
	# (STRATEGY-MODEL-CHECKER must be included, but things are more complex)
	else:
		# Like in the standard case, we should distinguish whether the module is a metamodule
		if case.metamodule is None:
			strategy_str = case.strategy_str
			# This program allows arbitrary strategy expressions, but the modelCheck
			# symbol expects the name of a declared strategy without arguments.
			# A module is issued to define it (hope there are no name conflicts).
			if '(' in strategy_str or strategy_str not in (s.getName() for s in case.module.getStrategies()):
				p.stdin.write(f'''smod MAIN-UMAUDEMC is
						protecting {str(case.module)} .
						strat %main-umaudemc% @ State .
						sd %main-umaudemc% := {strategy_str} .
					endsm\n'''.encode('utf-8'))
				strategy_str = '%main-umaudemc%'

			p.stdin.write(f"red modelCheck({case.initial_str}, {case.formula_str}, '{strategy_str}, {opaques}) .\n".encode('utf-8'))

		else:
			metalevel = maude.getModule('META-LEVEL')
			formula_term = case.metamodule.parseTerm(case.formula_str)
			p.stdin.write(f'''red in META-MODEL-CHECKER : metaModelCheck({case.metamodule_str},
							{metalevel.upTerm(case.initial)},
							{metalevel.upTerm(formula_term)},
							{metalevel.upStrategy(case.strategy)},
							{opaques}, true) .\n'''.encode('utf-8'))

	# Quit Maude after model checking
	p.stdin.write(b'quit .\n')
	p.stdin.flush()


def _memory_case(memusage, psutil, case, name, backend, args, log, maudebin=None):
	"""Measure memory usage of a single test case"""

	# If the name of a Maude binary has been given, we can measure the memory usage
	# for LTL properties directly using that binary (instead of within umaudemc).
	if name == 'maude' and maudebin is not None:
		stdin_origin = subprocess.PIPE
		call = [maudebin, case.filename]
	else:
		stdin_origin = subprocess.DEVNULL
		# Build the arguments of a call to umaudemc for this case
		call = _make_self_call(case, name)

	# Using Python library psutil
	if psutil is not None:
		# Call the program argument discarding its output
		p = psutil.Popen(call, stdin=stdin_origin, stdout=subprocess.DEVNULL)

		if stdin_origin == subprocess.PIPE:
			_feed_maude_binary(p, case)

		# The results of the two alternative methods memusage and psutil
		# do not usually coincide. Here the resident set size is observed.
		peak = 0
		total = 0
		last_heap = 0
		stack = 0

		# Regions in the memory map, if the [stack] path is present
		# we read it to get the stack memory used
		regions = [mm.path for mm in p.memory_maps()]

		if '[stack]' in regions:
			get_stack = lambda p: next((mm.rss for mm in p.memory_maps() if mm.path == '[stack]'), 0)
		else:
			get_stack = lambda p: 0

		while p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
			time.sleep(0.1)
			heap = p.memory_info().rss
			cstack = get_stack(p)
			for child in p.children(recursive=True):
				heap += child.memory_info().rss
				cstack += get_stack(child)

			# The peak of heap usage
			if heap > peak:
				peak = heap
			# The peak of stack usage
			if cstack > stack:
				stack = cstack
			# The positive variation of heap usage
			if heap > last_heap:
				total += heap - last_heap
			last_heap = heap

			if heap > 0:
				print(f'{tmn.clean_line}{name:10} {_print_size(heap):10} {_print_size(total):10} {_print_size(cstack):10}',
				      end='\r', flush=True)

		# It may happen that when arriving to the loop p is already over
		# and no relevant information is collected, but only in very small cases

		p.wait()

		# Separate the output of each case in a line
		if peak > 0:
			print()

	# Using memusage (which ignores children, i.e. it is unuseful for LTSmin and NuSMV)
	else:
		# The program is run under memusage
		p = subprocess.Popen([memusage] + call, stdin=stdin_origin, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

		if stdin_origin == subprocess.PIPE:
			_feed_maude_binary(p, case)

		p.wait()

		# Match the output of memusage to extract the figures
		m = MEMUSAGE_REGEX.search(p.stderr.read())

		if m is None:
			usermsgs.print_warning('Cannot understand the output of the memusage command. Ignoring case.')
			return

		total = int(m.group(1))
		peak = int(m.group(2))
		stack = int(m.group(3))
		print(f'{name:10} {_print_size(peak):10} {_print_size(total):10} {_print_size(stack):10}')

	log.writerow((case.rel_filename,
	              case.module,
	              case.initial_str,
	              case.formula_str,
	              case.strategy_str,
	              ' '.join(case.opaque),
	              case.expected,
	              case.ftype,
	              name,
	              total,
	              peak,
	              stack
	              ))


def memory_tests(suite, backends, args, only_file=None, only_logic=None,
                 out_file='memory.csv', repeats=None, resume=None):
	"""Measure memory usage by the model checkers with the test suite"""

	# There are two possibilities to measure memory usage and memusage
	# is preferred over psutil
	memusage = shutil.which('memusage')
	psutil = None

	if memusage is None or args.memory_method == 'psutil':
		try: import psutil
		except ImportError: pass

		if psutil is None:
			usermsgs.print_error("Either the psutil package or glibc's memusage are required "
			                     'to measure memory usage, but none can be found.\n'
			                     'The first can be installed with pip install psutil.')
			return 1
		else:
			memusage = None

	# Check whether a Maude binary has been selected (to measure memory usage
	# of the command instead of the maude backend)
	maudebin = os.getenv('MAUDE_BIN')

	if args.maudebin is not None:
		maudebin = args.maudebin

	if maudebin is not None and (not os.access(maudebin, os.X_OK) or not os.path.isfile(maudebin)):
		usermsgs.print_warning('The given Maude binary cannot be accessed or executed. Ignoring it.')
		maudebin = None

	# If continuing a previous execution, we append to the previous log
	open_mode = 'a' if resume.filename is not None and os.path.exists(out_file) else 'w'

	# A CSV file is written with a row for each test case and backend
	with open(out_file, open_mode) as logfile:
		log = csv.writer(logfile)

		if open_mode != 'a':
			log.writerow([
				'filename', 'module', 'initial', 'formula', 'strategy', 'opaque',
				'expected', 'logic', 'backend', 'total', 'peak', 'stack'
			])

		for source, cases in suite:
			if source is None:
				return 1

			# Ignore files that do not match pattern
			if only_file and not only_file.fullmatch(source):
				continue

			print(f'{tmn.bold}{source}{tmn.reset}')

			# Get track of the last executed case
			resume.filename = source

			for case, changed in cases:

				if case is None:
					return 1

				resume.case_number += 1

				# Ignore the formula if only-logic is present
				if only_logic and not case.ftype.lower() in only_logic:
					continue

				# Filter backends with the excluded attribute
				filtered_backends = _filter_backends(backends, case.exclude)

				for name, backend in filtered_backends:
					if case.ftype not in supported_logics[name]:
						continue

					resume.start()
					_memory_case(memusage, psutil, case, name, backend, args, log, maudebin)
					resume.stop()

			resume.case_number = 0


class RepeatFunc:
	"""Parse the repeats argument"""

	def __init__(self, text):
		#
		# The argument should be a list of integers separated by
		# colons, of the form
		#   times1:limit1:...:timesn:limitn:...[:timesn+1]
		# so that the number of repetitions is timesi for
		# the first i such that limiti is less than the
		# time argument.
		#
		tokens = list(map(int, text.split(':')))

		self.times = tokens[::2]
		self.limits = tokens[1::2]

		if len(self.times) == len(self.limits):
			self.times.append(1)

	def __call__(self, t):
		for i, lim in enumerate(self.limits):
			if t < lim:
				return self.times[i]

		return self.times[-1]


def _add_from(args, clue):
	"""Add the --from argument to a umaudemc invocation"""

	# Remove the previous --from from the list
	for i, arg in enumerate(args):
		# The option may have been given as --from followed
		# by another argument or as --from=value.
		if arg.startswith('--from'):
			if arg == '--from':
				args.pop(i+1)
			args.pop(i)

	args.append('--from')
	args.append(f'{clue.filename}:{clue.case_number}')

	return args


class ResumeClue:
	"""Hold the current position in the test suite to resume it"""

	def __init__(self, from_arg, timeout, restart=False):

		self.filename = None
		self.case_number = 0

		# Parse the --from argument (if present)
		if from_arg is not None:
			parts = from_arg.split(':')

			if len(parts) == 1:
				self.filename = parts[0]

			elif len(parts) == 2:
				self.filename = parts[0]
				self.case_number = int(parts[1])

			else:
				usermsgs.print_warning('The --from argument cannot be parsed. Ignoring it.')

		self.timeout = timeout
		self.restart = restart

		self.timer = None
		self.event = None
		self.started = False

		# The timeout is handled by a dedicated process with an event.
		# We could have used a threading.Timer (single use) or
		# signal.alarm (does not work in Windows) instead.
		if timeout:
			self.timer = threading.Thread(target=self.timeout_handler, daemon=True)
			self.event = threading.Event()
			self.timer.start()
		else:
			timeout = None

	def start(self):
		if self.timer is not None:
			self.event.set()

	def stop(self):
		if self.timer is not None:
			self.event.set()

	def timeout_handler(self):
		"""Handle timeouts in a separate thread"""

		while True:
			# Wait for the start token...
			self.event.wait()
			self.event.clear()
			# ...and then for the stop token or the timeout
			# (and a second more to allow backends to handle the timeouts
			# themselves, usually in cleaner way than here)
			if not self.event.wait(timeout=self.timeout+1):
				# When the timeout is reached and the restart flag is set, we execute the
				# test subcommand again from the next test case.
				if self.restart:
					print('Execution timed out. Restarting...')

					# This solution is tricky and dirty, because it consists on replacing
					# the current umaudemc process by a new one that starts from the case
					# indicated by the --from argument. In case of backends that invoke
					# external programs, these would not be terminated, so they should
					# handle timeouts themselves. Just in case, we also try it here.
					try:
						from psutil import Process
						for child in Process().children():
							child.kill()
					except:
						pass

					os.execv(sys.executable, [sys.executable, '-m', 'umaudemc'] + _add_from(sys.argv[1:], self))

				# Otherwise, we simply write the command to continue from that case
				else:
					print('Execution timed out. It can be resumed from the next case with:')
					print('  python -m umaudemc ' + ' '.join(_add_from(sys.argv[1:], self)))
					os._exit(1)

			self.event.clear()


def test(args):
	"""Entry point for the test subcommand"""

	maude.init(advise=args.advise)

	# The test function is selected depending on the -- flags
	test_function = run_tests
	resume = None

	try:
		# These are filter that limit which case are checked based on filenames or logics
		only_file = re.compile(args.only_file) if args.only_file else None
		only_logic = args.only_logic.lower().split(',') if args.only_logic else None

		# Warn whether some backends are not available
		backends, unavailable = get_backends(args.backend)

		if len(unavailable) > 0:
			usermsgs.print_warning('The following backends have not been found: '
			                       + ' '.join((name for name, backend in unavailable)))

		# The resume object holds the filename and index of the last case run,
		# so the execution can be resumed from this point. It also serves as
		# the timer for timeouts.
		resume = ResumeClue(args.fromcase, args.timeout, args.resume)

		if args.benchmark:
			if args.memory:
				usermsgs.print_warning('Flag --memory has been ignored since --benchmark is present too.')

			output_file = 'test.csv'
			test_function = benchmark_tests

		elif args.memory:

			output_file = 'memory.csv'
			test_function = memory_tests

		else:
			output_file = None
			test_function = run_tests

		# Read the test suite from the given file
		test_suite = read_suite(args.file, from_file=resume.filename, skip_case=resume.case_number)

		return test_function(test_suite, backends, args, only_file, only_logic,
		                     out_file=output_file,
		                     repeats=RepeatFunc(args.repeats),
		                     resume=resume)

	except FileNotFoundError as fnfe:
		usermsgs.print_error('File not found: ' + str(fnfe) + '.')

	except re.error as rerr:
		usermsgs.print_error('Bad regular expression passed to --only-file: ' + str(rerr) + '.')
		return 1

	except KeyboardInterrupt:
		if resume is not None:
			print('\nOperation interrupted by the user. It can be resumed from the next case with:')
			print('  python -m umaudemc ' + ' '.join(_add_from(sys.argv[1:], resume)))

	return 1
