#
# LTSmin backend for umaudemc
#

import os
import re
import shutil
import subprocess
import sys
import tempfile

from .. import usermsgs
from ..common import default_model_settings
from ..formulae import collect_aprops

# Operators that need a special treatment
# (because they contain arguments that are not formulae)
_special_ops = {'<_>_', '`[_`]_', 'mu_._', 'nu_._'}

# Regex to remove ANSI escape codes when printing LTSmin output
_ansi_escape = re.compile(rb'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')


# Translation tables for the different logics supported by LTSmin
# (the second column is the operator's priority)

_ltl_translation = {
	'True'		: ('true',	0),
	'False'		: ('false', 	0),
	'~_'		: ('! {}',	2),
	'`[`]_'		: ('[] {}',	3),
	'<>_'		: ('<> {}',	3),
	'O_'		: ('X {}',	3),
	'_/\\_'		: ('{} && {}',	4),
	'_\\/_'		: ('{} || {}',	5),
	'_<->_'		: ('{} <-> {}',	6),
	'_->_'		: ('{} -> {}',	7),
	'_U_'		: ('{} U {}',	8),
	'_R_'		: ('{} R {}',	8)
}

_ctl_translation = {
	'True'		: ('true',	0),
	'False'		: ('false', 	0),
	'~_'		: ('! {}',	2),
	'E_'		: ('E {}',	3),
	'A_'		: ('A {}',	3),
	'`[`]_'		: ('[] {}',	3),
	'<>_'		: ('<> {}',	3),
	'O_'		: ('X {}',	3),
	'_/\\_'		: ('{} && {}',	4),
	'_\\/_'		: ('{} || {}',	5),
	'_<->_'		: ('{} <-> {}',	6),
	'_->_'		: ('{} -> {}',	7),
	'_U_'		: ('{} U {}',	8)
}

_mu_translation = {
	'True'		: ('true',		0),
	'False'		: ('false',		0),
	'~_'		: ('! {}',		2),
	'_/\\_'		: ('{} && {}',		3),
	'_\\/_'		: ('{} || {}',		4),
	'<.>_'		: ('E X {}',		5),
	'`[.`]_'	: ('A X {}',		5),
	'mu_._'		: ('mu {} . {}',	6),
	'nu_._'		: ('nu {} . {}',	6)
}

# Although the ltsmin-mucalc manpage say that the priority
# of the conjunction is higher than that of the disjunction,
# this does not work in reality

_mucalc_translation = {
	'True'		: ('true',		0),
	'False'		: ('false',		0),
	'~_'		: ('! {}',		1),
	'_/\\_'		: ('{} && {}',		2),
	'_\\/_'		: ('{} || {}',		2),
	'nu_._'		: ('nu {} . {}',	4),
	'mu_._'		: ('mu {} . {}',	5),
	'`[.`]_'	: (None,		6),
	'`[_`]_'	: ('[ "{}" ] {}',	6),
	'<.>_'		: (None,		7),
	'<_>_'		: ('< "{}" > {}',	7),
}

# Arbitrary atomic propositions in Maude may contain characters not admitted
# in LTSmin state label identifiers (parentheses, commas, space...) unless
# escaped with a backslash. According to ltsmin-lib/ltsmin-syntax.c, these are
# the admitted alphabet and the list of keywords that must also be escaped.

_ltsmin_keyword = {'begin', 'end', 'state', 'edge', 'init', 'trans', 'sort', 'map'}
_admitted_chars = frozenset('_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890')


def _escape_to_admitted(aprop):
	"""Escape characters not allowed in an LTSmin identifier"""
	s = list(aprop)
	for i, c in enumerate(s):
		if c not in _admitted_chars:
			s[i] = '\\' + c
	return ''.join(s)


def _translate_aprop(aprop):
	"""Translate an atomic proposition to the format admitted by LTSmin"""

	# According to ltsmin-lib/ltsmin-syntax.c's fprint_ltsmin_ident function
	return ('\\' + aprop) if aprop in _ltsmin_keyword else _escape_to_admitted(aprop)


def _translate_label(label):
	"""Translate a rule or strategy label as action label"""

	# It may be an ['opaque', name] list or simply a rule name
	return f'opaque_{label[1]}' if label[0] == 'opaque' else label


def _has_edge_labels(form):
	"""Check whether the formula contains edge labels"""
	head, *rest = form

	if head in {'Prop', 'Var'}:
		return False
	elif head in {'<_>_', '`[_`]_'}:
		return True
	else:
		return any(map(_has_edge_labels, rest))


def _preprocess_formula(formula, skip=()):
	"""Preprocess a formula to remove unsupported operators and simplify others"""
	head, *rest = formula

	if head in {'Prop', 'Var'}:
		return formula
	elif head == '_R_' and rest[0] == ['False']:
		return ['`[`]_', _preprocess_formula(rest[1], skip)]
	elif head == '_R_' and '_R_' in skip:
		left  = ['~_', _preprocess_formula(rest[1], skip)]
		right = ['~_', _preprocess_formula(rest[0], skip)]
		return ['~_', ['_U_', left, right]]
	elif head == '_U_' and rest[0] == ['True']:
		return ['<>_', _preprocess_formula(rest[1], skip)]
	elif head in {'`[_`]_', '<_>_'}:
		return formula
	else:
		return [head] + [_preprocess_formula(arg, skip) for arg in rest]


def _special_mu(form, trsymb, prio):
	"""Handle the special operators for the LTSmin's mu syntax (mu, nu)"""
	head, var, arg = form
	return trsymb.format(var[1], _make_ltsmin_formula(arg, _mu_translation,
	                                                  prio, _special_mu))


def _special_mucalc(form, trsymb, prio):
	"""Handle the special operators for the LTSmin's mucalc syntax (mu, nu, <_> and [_])"""
	head, *rest = form

	if head in {'mu_._', 'nu_._'}:
		return trsymb.format(rest[0][1],
		                     _make_ltsmin_formula(rest[1], _mucalc_translation, prio, _special_mucalc))
	elif head in ['<_>_', '`[_`]_']:
		# LTSmin's mucalc formulae only admit one action per modality,
		# so a conjunction or disjunction has to be built to support multiple actions.
		connector = ' || ' if head == '<_>_' else ' && '
		child = _make_ltsmin_formula(rest[1], _mucalc_translation, prio, _special_mucalc)
		# Parentheses are added when logical connectors are introduced (the priority changes)
		if len(rest[0]) > 1:
			trsymb = '(' + trsymb + ')'
		clauses = [trsymb.format(_translate_label(action), child) for action in rest[0]]
		return connector.join(clauses)


def _make_ltsmin_ltl(form):
	return _make_ltsmin_formula(form, _ltl_translation)


def _make_ltsmin_ctl(form):
	return _make_ltsmin_formula(form, _ctl_translation)


def _make_ltsmin_mu(form):
	return _make_ltsmin_formula(form, _mu_translation,
	                            special_op=_special_mu)


def _make_ltsmin_mucalc(form, labels):
	# The translation of the modalities with dot (no matter which action is
	# chosen) requires explicitly repeating the operator for each label in
	# the module. They are directly inserted in the translation table.
	_mucalc_translation['<.>_'] = ' || '.join(f'(< "{label}" > {{0}})' for label in labels), 7
	_mucalc_translation['`[.`]_'] = ' && '.join(f'([ "{label}" ] {{0}})' for label in labels), 6

	return _make_ltsmin_formula(form, _mucalc_translation,
	                            special_op=_special_mucalc)


def _make_ltsmin_formula(form, translation, out_prio=80, special_op=None):
	head, *rest = form

	if head == 'Prop':
		translated = _translate_aprop(rest[0])
		# In the mucalc formulae, atomic propositions use a different syntax
		# This only work (with state labels in addition to state variables)
		# with https://github.com/utwente-fmt/ltsmin/pull/185
		if special_op == _special_mucalc:
			result, prio = f'{{ {translated} = 1 }}', 1
		else:
			result, prio = _translate_aprop(rest[0]) + ' == true', 1
	elif head == 'Var':
		result, prio = rest[0], 0
	else:
		trsymb, prio = translation[head]

		# Special case for mu/nu operators
		if head in _special_ops:
			result = special_op(form, trsymb, prio)

		else:
			newargs = [_make_ltsmin_formula(f, translation, prio, special_op) for f in rest]
			result = trsymb.format(*newargs)

	return '(' + result + ')' if out_prio <= prio else result


class LTSmin:
	"""LTSmin backend (session creator)"""

	def __init__(self):
		# Dictionary of LTSmin commands paths
		self.pins2lts = {}
		# Path of the Maude plugin
		self.maudemc = None
		# Paths of other related programs
		self.ltsmin_convert = None
		self.pbespgsolve = None

	def find(self, with_mucalc=False):
		"""Find the LTSmin binary and Maude module"""
		return self.find_ltsmin() and self.find_maudemc() and (not with_mucalc or self.pbespgsolve)

	@property
	def module_suffix(self):
		"""Language module suffix for the host platform"""
		if sys.platform == 'darwin':
			return '.dylib'
		elif sys.platform in {'win32', 'cygwin'}:
			return '.dll'
		else:
			return '.so'

	def find_maudemc(self, maudemc_path=os.getenv('MAUDEMC_PATH')):
		"""Tries to find the Maude plugin for LTSmin"""
		if maudemc_path is None:
			maudemc_path = os.path.realpath('libmaudemc' + self.module_suffix)

		if os.path.isfile(maudemc_path) and os.access(maudemc_path, os.X_OK):
			self.maudemc = maudemc_path
			return True

		return False

	def find_ltsmin(self, ltsmin_path=os.getenv('LTSMIN_PATH')):
		"""Find LTSmin binaries"""
		if ltsmin_path is not None:
			ltsmin_paths = ltsmin_path.split(os.pathsep)

			for path in ltsmin_paths:
				self.find_command(path, 'seq')
				self.find_command(path, 'sym')

		# LTSmin path is not given, but maybe it is in the path
		if len(self.pins2lts) == 0:

			for variant in ['seq', 'sym']:
				path = shutil.which('pins2lts-{}'.format(variant))
				if path is not None:
					self.pins2lts[variant] = path

		if len(self.pins2lts) == 0:
			return False

		# Find ltsmin-convert y pbespgsolve, needed for mucalc
		if ltsmin_path is not None:
			self.ltsmin_convert = self.find_tool(ltsmin_paths, 'ltsmin-convert')
			if self.ltsmin_convert is not None:
				self.pbespgsolve = self.find_tool(ltsmin_paths, 'pbespgsolve')

		if self.ltsmin_convert is None:
			self.ltsmin_convert = shutil.which('ltsmin-convert')
		if self.ltsmin_convert is not None and self.pbespgsolve is None:
			self.pbespgsolve = shutil.which('pbespgsolve')

		return True

	def find_tool(self, paths, name):
		"""Find a specific tool in the given list of paths"""
		for path in paths:
			cmd = os.path.join(path, name)
			if os.path.isfile(cmd) and os.access(cmd, os.X_OK):
				return cmd

		return shutil.which(name)

	def find_command(self, path, variant):
		"""Find a LTSmin command in the given path"""
		if self.pins2lts.get(variant) is None:
			cmd = os.path.join(path, 'pins2lts-' + variant)
			if os.path.isfile(cmd) and os.access(cmd, os.X_OK):
				self.pins2lts[variant] = cmd

	def new_runner(self):
		"""Create a new instance or session of LTSmin"""
		return LTSminRunner(self)

	def check(self, filename=None, module_str=None, metamodule_str=None, labels=(),
	          formula=None, logic=None, aprops=None, full_matchrew=False, opaque=(),
	          strategy_str=None, term_str=None, merge_states='default',
	          purge_fails='default', timeout=None, extra_args=(), verbose=False, **_):
		"""Check a model-checking problem directly with LTSmin"""

		purge_fails, merge_states = default_model_settings(logic, purge_fails, merge_states,
		                                                   strategy_str, tableau=False)

		if aprops is None:
			aprops = set()
			collect_aprops(formula, aprops)

		# Create an instance of LTSmin and run it
		runner = self.new_runner()
		runner.set_module(module_str, metamodule_str, labels, opaque)
		runner.add_formula(formula, logic, aprops)

		return runner.run(filename, term_str, strategy_str,
		                  biased_matchrew=not full_matchrew,
		                  opaque_strats=opaque,
		                  extra_args=extra_args,
		                  purge_fails=purge_fails == 'yes',
		                  merge_states=merge_states,
		                  verbose=verbose,
		                  timeout=timeout)


class LTSminRunner:
	"""Runs LTSmin"""

	STATS_REGEX = re.compile(br'maude-mc: (\d+) system states explored(?: \((\d+) real\))?, (\d+) rewrites')
	RESULT_REGEX = re.compile(br'^pins2lts-[^:]+: Formula')
	BUCHI_REGEX = re.compile(br'^pins2lts-seq: buchi has (\d)+ states')
	CONVERT_REGEX = re.compile(br'^ltsmin-convert: Number of states: (\d+)')

	def __init__(self, installation):
		# LTSmin installation
		self.ltsmin = installation
		# The name of the module and the term of the metamodule
		# (if metamodule is present, module is the name of the
		#  module where the metamodule is expressed)
		self.module = None
		self.metamodule = None
		# Labels of the rules in the module
		self.labels = None
		# List of formulae to be checked (only one is supported now)
		self.formulae = []
		# Type of the formulae
		self.ftype = None
		self.aprops = set()

	def set_module(self, module, metamodule=None, labels=(), opaque_strats=()):
		"""
		Set the Maude module of the model-checking problem.
		:param module: Name of the module.
		:type module: str
		:param metamodule: Module metarepresentation as a string (if not null,
		  the previous module argument refers to the module where this metamodule
		  is expressed)
		:type metamodule: str
		:param labels: Edge labels of that module.
		:type labels: collection of str
		:param opaque_strats: Opaque strategy names.
		:type opaque_strats: collection of str
		:returns: r:...
		"""
		self.module = module
		self.metamodule = metamodule
		self.labels = labels + [f'opaque_{name}' for name in opaque_strats]

	def add_formula(self, formula, ftype, aprops):
		"""
		Add a temporal formula to be checked.
		:param formula: Formula as a nested list structure.
		:type formula: list
		:param ftype: Temporal logic (CTL, CTL*, LTL, propLogic, or Mucalc)
		:type ftype: str
		:param aprops: Atomic propositions present in the formula (as strings)
		:type aprops: collection of str
		"""
		self.aprops.update(aprops)
		ltsmin_formula = ''

		formula = _preprocess_formula(formula, [] if ftype == 'LTL' else ['_R_'])

		if ftype == 'CTL':
			self.ftype = 'ctl'
			ltsmin_formula = _make_ltsmin_ctl(formula)
		elif ftype == 'CTL*':
			self.ftype = 'ctl-star'
			ltsmin_formula = _make_ltsmin_ctl(formula)
		elif ftype == 'LTL':
			self.ftype = 'ltl'
			ltsmin_formula = _make_ltsmin_ltl(formula)
		elif ftype == 'propLogic':
			self.ftype = 'ltl'
			ltsmin_formula = _make_ltsmin_ltl(formula)
		elif ftype == 'Mucalc':
			if _has_edge_labels(formula):
				self.ftype = 'mucalc'

				# Edge and state labels cannot be used at the same time in LTSmin v3.0.2
				# nor in the current repository version. A patch has been proposed to
				# solve, but it has not currently been accepted or merged. However, it can
				# be used if built or downloaded from https://github.com/fadoss/ltsmin.

				if len(aprops) > 0:
					usermsgs.print_warning(
						'Using both edge and state labels may not work in your version of LTSmin.\n'
						'(see https://github.com/utwente-fmt/ltsmin/pull/185).')
				ltsmin_formula = _make_ltsmin_mucalc(formula, self.labels)
			else:
				self.ftype = 'mu'
				ltsmin_formula = _make_ltsmin_mu(formula)
		else:
			usermsgs.print_error('Internal error in LTSmin connection.')

		self.formulae.append(ltsmin_formula)

	def _read_pins2lts_output(self, output):
		"""Read the model checking result for pins2lts tools output"""
		holds = None
		stats = {}

		# The statistical information and the model-checking result are
		# extracted using regular expressions and looking for specific
		# messages. The witnesses for the latter depend on the logic.

		witness = {'ctl': 0, 'ctl-star': 0, 'mu': 0, 'ltl': 1, 'mucalc': 2}[self.ftype]

		for line in output.splitlines():
			match = self.STATS_REGEX.match(line)
			if match:
				stats['states'] = int(match.group(1))
				stats['real_states'] = int(match.group(2)) if match.group(2) else None
				stats['rewrites'] = int(match.group(3))
			elif witness == 0:
				if self.RESULT_REGEX.match(line):
					holds = line.find(b'does not hold') < 0
			elif witness == 1:
				if line.startswith(b'pins2lts-seq: Accepting cycle FOUND!'):
					holds = False
				elif line.startswith(b'pins2lts-seq: Empty product with LTL!'):
					holds = True
				else:
					match = self.BUCHI_REGEX.match(line)
					if match:
						stats['buchi'] = int(match.group(1))

		return holds, stats

	def _read_convert_output(self, output, stats):
		"""Read output (number of game states) from the ltsmin-convert tool"""
		for line in output.splitlines():
			match = self.CONVERT_REGEX.match(line)
			if match:
				stats['game'] = int(match.group(1))

	def run(self, filename, initial, strategy, biased_matchrew=True, opaque_strats=(),
	        merge_states='state', purge_fails=True, extra_args=(), raw=False, verbose=False,
	        no_advise=False, depth=0, timeout=None):
		"""Run LTSmin to model check temporal formulae"""

		if merge_states == 'no':
			merge_states = 'none'

		# Arguments to be passed to LTSmin
		args = list(extra_args) + [
			'--loader=' + self.ltsmin.maudemc,
			'--initial=' + initial,
			'--merge-states=' + merge_states,
			filename,
		]

		# The formulae to be checked
		args += [f'--{self.ftype}={formula}' for formula in self.formulae]

		# Other Maude model input data

		if self.module is not None:
			args.append('--module=' + self.module)

		if self.metamodule is not None:
			args.append('--metamodule=' + self.metamodule)

		if len(self.aprops) > 0:
			args.append('--aprops=' + ','.join(prop for prop in self.aprops)),

		if biased_matchrew:
			args.append('--biased-matchrew')

		if opaque_strats:
			args.append('--opaque-strats=' + ','.join(opaque_strats))

		if strategy is not None:
			args.append('--strat=' + strategy)

		if purge_fails:
			args.append('--purge-fails')

		if no_advise:
			args.append('--no-advise')

		# LTSmin-specific configuration for each logic

		# When model checking with strategies the ltsmin or textbook LTL semantics
		# is required, since otherwise self-loops will be added to deadlock states.
		if self.ftype == 'ltl':
			args.append('--ltl-semantics=' + ('spin' if strategy is None else 'ltsmin'))

		# For the mucalc logic, we need to export the processed model to
		# a directory for later processing (parity game solving).
		if self.ftype == 'mucalc':
			tempdir = tempfile.mkdtemp(suffix='.dir')
			args.append(tempdir)

		# The symbolic tool of LTSmin is used for ctl, ctlstar and mu only
		variant = 'sym' if self.ftype.startswith('ctl') or self.ftype == 'mu' else 'seq'

		# Depth bounds are only allowed with the sequential model checker.
		if variant == 'seq' and depth > 0:
			args.append(f'--max={depth}')

		# print('\033[33m', self.ltsmin.pins2lts[variant], ' ', ' '.join(args), '\033[0m', sep='')

		# Add the directory of the Maude file being used to the MAUDE_LIB
		# environment variable so that dependencies can be found with relative paths.
		# Moreover, MAUDE_LIB_LTSMIN is preferred over MAUDE_LIB to cover the case
		# when the LTSmin plugin and the maude Python library have been built with
		# incompatible Maude versions.

		maude_lib_ltsmin = os.getenv('MAUDE_LIB_LTSMIN')
		new_maude_lib = os.getenv('MAUDE_LIB', '') if maude_lib_ltsmin is None else maude_lib_ltsmin
		new_maude_lib = ('' if new_maude_lib == '' else new_maude_lib + os.pathsep) \
		                + os.path.dirname(filename)

		if raw:
			os.execve(self.ltsmin.pins2lts[variant], [self.ltsmin.pins2lts[variant]] + args,
			          env=dict(os.environ, MAUDE_LIB=new_maude_lib))
		else:
			# Run the LTSmin tool with the arguments prepared above
			try:
				status = subprocess.run([self.ltsmin.pins2lts[variant]] + args, capture_output=True,
				                        env=dict(os.environ, MAUDE_LIB=new_maude_lib), timeout=timeout)

			except subprocess.TimeoutExpired:
				usermsgs.print_error(f'LTSmin execution timed out after {timeout} seconds.')
				return (None,) * 2

			if verbose:
				print(status.stderr[:-1].decode('utf-8'))

			# 1 is returned when there is an error or when pins2lts-sym finds a counterexample
			if status.returncode > 1:
				usermsgs.print_error('An error has been produced while running LTSmin:\n'
				                     + _ansi_escape.sub(b'', status.stderr[:-1]).decode('utf-8'))
				return (None,) * 2

			holds, stats = self._read_pins2lts_output(status.stderr)

			if holds is None and self.ftype != 'mucalc':
				usermsgs.print_error('An error has been produced while running LTSmin:\n'
				                     + _ansi_escape.sub(b'', status.stderr[:-1]).decode('utf-8'))
				return (None,) * 2

			if self.ftype == 'mucalc':
				if self.ltsmin.pbespgsolve is None:
					usermsgs.print_warning(
						'The parity solver pbespgsolve is required to process the output, but it cannot be found.\n'
						'It can be downloaded as part of mCRL2 from https://www.mcrl2.org.\n'
						'The intermediate output files are left in ' + tempdir)
					return (None,) * 2

				# Convert the dir format output from pins2lts-seq to a parity game that can be
				# solved by pbespgsolve from mCRL2. pins2lts-sym can directly solve parity games
				# with the --pg-solve option, but there is bug in its mucalc implementation that
				# make it return wrong answers (https://github.com/utwente-fmt/ltsmin/issues/184).

				pgname = os.path.join(tempdir, 'game.pg')
				convert_status = subprocess.run((self.ltsmin.ltsmin_convert, '--rdwr', tempdir, pgname), capture_output=True)
				if convert_status.returncode != 0:
					usermsgs.print_error('Error while generating the parity game with ltsmin-convert:'
					                     + convert_status.stderr.decode('utf-8'))
					return (None,) * 3

				self._read_convert_output(convert_status.stderr, stats)
				if verbose:
					print(convert_status.stderr[:-1].decode('utf-8'))

				# Solve the game with pbespgsolve, whose output is simply true or false

				solver_status = subprocess.run((self.ltsmin.pbespgsolve, '-ipgsolver', pgname), capture_output=True)
				shutil.rmtree(tempdir)

				if solver_status.returncode == 0:
					holds = solver_status.stdout.startswith(b'true')

			return holds, stats
