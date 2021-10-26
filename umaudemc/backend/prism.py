#
# PRISM probabilistic backend for umaudemc
#

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from .. import probabilistic as pbs
from ..common import usermsgs
from ..wrappers import create_graph

# Result line printed by PRISM
_RESULT_REGEX = re.compile(b'^Result: ([\\d.E-]+|Infinity|false|true)(.*)')
# Additional information on the result
_MORE_RESULT_REGEX = re.compile(b' \\(\\+/- ([\\d.E-]+) .*; rel err ([\\d.E-]+)')

# Translation of LTL, CTL and PCTL formulae

_translation = {
	'True':   ('true',        0, False),
	'False':  ('false',       0, False),
	'~_':     ('! ({})',      1, False),
	'_/\\_':  ('{} & {}',     2, False),
	'_\\/_':  ('{} | {}',     3, False),
	'_<->_':  ('{} <=> {}',   4, False),
	'_->_':   ('{} => {}',    5, False),
	'_U_':    ('{} U {}',     6, False),
	'_U__':   ('{} U{b} {}',  6, True),
	'_R_':    ('{} R {}',     6, False),
	'_R__':   ('{} R{b} {}',  6, True),
	'_W_':    ('{} W {}',     6, False),
	'_W__':   ('{} W{b} {}',  6, True),
	'`[`]_':  ('G {}',        7, False),
	'`[`]__': ('G{b} {}',     7, True),
	'<>_':    ('F {}',        7, False),
	'<>__':   ('F{b} {}',     7, True),
	'O_':     ('X {}',        7, False),
	'A_':     ('A [ {} ]',    8, False),
	'E_':     ('E [ {} ]',    8, False),
	'P__':    ('P{b} [ {} ]', 9, True),
}


def _preprocess_formula(formula):
	"""Preprocess the formula, simplifying some operators"""
	head, *rest = formula

	if head == 'Prop':
		return formula
	elif head == '_R_' and rest[0] == ['False']:
		return ['`[`]_', _preprocess_formula(rest[1])]
	elif head == '_U_' and rest[0] == ['True']:
		return ['<>_', _preprocess_formula(rest[1])]
	elif head in ('<=', '<', '>=', '>', '-'):
		return formula
	else:
		return [head] + [_preprocess_formula(arg) for arg in rest]


def _make_bound(bound):
	"""Format bound to be attached to a PRISM operator"""

	if bound[0] != '-':
		return bound[0] + str(bound[1])
	else:
		return f'[{bound[1]}, {bound[2]}]'


def _translate_aprop(aprop):
	"""Translate an atomic proposition to a PRISM label"""
	return aprop.replace('"', '\\"').replace('(', '').replace(')', '')


def _make_prism_formula(form, translation, out_prio=20):
	"""Translate a formula to the PRISM format"""
	head, *rest = form

	if head == 'Prop':
		result, prio = '"{}"'.format(_translate_aprop(rest[0])), 0
	else:
		trans, prio, has_bound = translation[head]
		# Bound annotation are always the first entry in the argument list
		if has_bound:
			bound = _make_bound(rest[0])
			rest.pop(0)
		else:
			bound = None
		result = trans.format(*[_make_prism_formula(arg, translation, prio) for arg in rest], b=bound)

	return '(' + result + ')' if out_prio <= prio else result


def make_prism_formula(form):
	"""Translate a formula to the PRISM format"""
	return _make_prism_formula(_preprocess_formula(form), _translation)


def _collect_raw_aprops(form, aprops):
	"""Collect atomic propositions from raw formula"""

	# The start of the last string, whether a escape character
	# has been just seen, the algorithm state (out, string or braces),
	# and the position where the current proposition starts
	last, escaped, state, pos = 0, False, 'o', 0

	# Walk through the formula accumulating the strings
	# found, unless they are written between curly braces
	for k, c in enumerate(form):
		if c == '"' and not escaped and state != 'b':
			if state == 'o':
				state, pos = 's', k+1
			else:
				state = 'o'
				aprops.add(form[pos:k])

		# Reward names are written within curly braces
		elif c == '{' and state == 'o':
			state = 'b'

		elif c == '}' and state == 'b':
			state = 'o'

		# Whether the next character is escaped
		escaped = not escaped and (c == '\\')


class PRISMBasedBackend:
	"""PRISM-like backend connector"""

	def __init__(self):
		self.command = None

	def run(self, graph, form, ftype, aprops, extra_args=(), raw=False, cost=False,
	        formula_str=None, timeout=None, distrib=None, reward=None):
		"""Run PRISM to solve the given problem"""

		# If PRISM has not been found, end with None
		if self.command is None:
			return None, None

		# Translate the Maude-parsed formula (if not raw)
		prism_formula = make_prism_formula(form) if form is not None else formula_str

		# The model is written to a temporary file so that it can be read by PRISM.
		# The procedure is explained in the NuSMV backend source.

		with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
			grapher = PRISMGenerator(tmpfile, aprops=aprops, distrib=distrib)

			# Output the DTMC or MDP for the model
			# Expand the graph and calculate the output degrees
			grapher.graph(graph)
			# Output the map of an user-specified reward
			if reward is not None:
				grapher.make_reward(graph, 'user', reward)

			tmpfile.flush()

		# Whether probability or rewards are to be calculated
		query = 'R' if cost else ('P' if reward is None else 'R{"user"}')

		# A raw formula has been given
		if form is None:
			full_formula = formula_str

		# If the model is a DTMC, we calculate the probability
		# (or reward expectation) that the property holds
		elif not graph.nondeterminism:
			# However, if the head of the formula is a P operator,
			# we prefer a Boolean result
			if form[0] == 'P__':
				full_formula = prism_formula
			else:
				full_formula = f'{query}=? [ {prism_formula} ]'

		# Otherwise, if the model is a MDP, we calculate the minimum
		# and maximum probabilities for the formula
		else:
			# Multiple formulae are admitted separated by semicolon
			full_formula = f'{query}min=? [ {prism_formula} ] ; {query}max=? [ {prism_formula} ]'

		# Record the time when the actual backend has run for statistics
		start_time = time.perf_counter_ns()

		# The backend's command is called on the generated file
		result, stats = self.run_command(tmpfile.name, full_formula, extra_args, timeout, raw), None

		if result is not None:
			stats = {
				'states': len(graph),
				'rewrites': grapher.getNrRewrites() + graph.getNrRewrites(),
				'backend_start_time': start_time,
				'real_states': graph.getNrRealStates()
			}

		os.remove(tmpfile.name)
		return result, stats

	def check(self, graph=None, module=None, formula=None, extra_args=(), get_graph=False,
	          timeout=None, cost=False, formula_str=None, aprops=None, logic=None,
	          dist=None, reward=None, **kwargs):
		"""Solves a model-checking problem with PRISM"""

		# Create the graph if not provided by the caller
		if graph is None:
			graph = pbs.AssignedGraph(create_graph(logic=logic, tableau=True, **kwargs), dist)
			graph.expand()

		# Extract the atomic proposition term from the raw formulae
		if formula is None:
			aprops = set()
			_collect_raw_aprops(formula_str, aprops)

		# Parse the atomic proposition to be evaluated on the states
		aprop_terms = [module.parseTerm(prop) for prop in aprops]

		holds, stats = self.run(graph, formula, logic, aprop_terms, extra_args, cost=cost,
		                        formula_str=formula_str, timeout=timeout, distrib=dist,
		                        reward=reward)

		if holds is not None and get_graph:
			stats['graph'] = graph

		return holds, stats


class PRISMBackend(PRISMBasedBackend):
	"""PRISM backend connector"""

	def find(self):
		"""Tries to find PRISM"""

		if os.getenv('PRISM_PATH') is not None:
			prism_path = os.path.join(os.getenv('PRISM_PATH'), 'prism')

			if os.path.isfile(prism_path) and os.access(prism_path, os.X_OK):
				self.command = prism_path

		# Look for it in the system path
		if not self.command:
			self.command = shutil.which('prism')

		return self.command is not None

	def run_command(self, filename, formula, extra_args, timeout, raw):
		"""Run PRISM and parse the result"""

		try:
			status = subprocess.run([self.command, filename, '-pf', formula] + list(extra_args),
			                        capture_output=not raw, timeout=timeout)

		except subprocess.TimeoutExpired:
			# PRISM can handle timeouts by itself
			usermsgs.print_error(f'PRISM execution timed out after {timeout} seconds.')
			return None

		# print(status.stdout.decode('utf-8'))

		if status.returncode != 0:
			usermsgs.print_error('An error was produced while running PRISM:\n'
			                     + status.stdout[:-1].decode('utf-8'))
			return None

		# Parse the PRISM output to obtain the result
		result = None

		for line in status.stdout.splitlines():
			match = _RESULT_REGEX.match(line)
			# The result can be true, false, or a probability
			if match:
				token = match.group(1)
				extra = match.group(2)

				if token == b'true':
					result = pbs.QuantitativeResult.make_boolean(True)
				elif token == b'false':
					result = pbs.QuantitativeResult.make_boolean(False)
				else:
					# Additional information (relative error, ...) is parsed too
					match = _MORE_RESULT_REGEX.match(extra)
					extra = {}

					if match:
						extra['abs_error'] = float(match.group(1))
						extra['rel_error'] = float(match.group(2))

					# If this result is not the first one, we consider it is
					# range (for the MDP case). Raw formulae may produce more
					# results, but this is not supported.
					if result is None:
						result = pbs.QuantitativeResult.make_number(float(token), extra)
					else:
						result = pbs.QuantitativeResult.make_range(result, float(token), extra)

			# Errors are printed by PRISM on the standard output
			if line.startswith(b'Error: '):
				usermsgs.print_error('Error: ' + line[7:].decode('utf-8'))

		return result


class PRISMGenerator:
	"""Generator of PRISM models"""

	def __init__(self, outfile=sys.stdout, aprops=(), distrib=None):
		self.visited = set()
		self.aprops = aprops
		self.outfile = outfile

		self.satisfies = None
		self.true_term = None

		# Number of rewrites used to check atomic propositions
		self.nrRewrites = 0

	def getNrRewrites(self):
		return self.nrRewrites

	def check_aprop(self, graph, propNr, stateNr):
		"""Check whether a given atomic proposition holds in a state"""
		t = self.satisfies.makeTerm((graph.getStateTerm(stateNr), self.aprops[propNr]))
		self.nrRewrites += t.reduce()
		return t == self.true_term

	def graph(self, graph):
		"""Generate a PRISM input file"""

		# Find the satisfaction (|=) symbol and the true constant to be used
		# when testing atomic propositions.

		module = graph.getStateTerm(0).symbol().getModule()
		boolkind = module.findSort('Bool').kind()

		self.satisfies = module.findSymbol('_|=_', (module.findSort('State').kind(),
							    module.findSort('Prop').kind()),
		                                   boolkind)

		self.true_term = module.findSymbol('true', (), boolkind).makeTerm(())

		# Build the model specification in the PRISM format
		model_type = 'mdp' if graph.nondeterminism else 'dtmc'

		# A discrete-time Markov chain or Markov decision process module is created
		print(f'{model_type}\n\nmodule M\n\tx : [0..{len(graph)}] init 0;\n', file=self.outfile)

		# For each reachable state, we first write the transition relation or distribution
		for state, children in graph.transitions():
			# If the state does not have successors, we just continue
			# (the stuttering-extension of deadlock states is done by PRISM)

			update = ' + '.join(f"{p}:(x'={nexts})" for p, nexts in children)
			print(f'\t[] x={state} -> {update};', file=self.outfile)

		print('\nendmodule', file=self.outfile)

		# Define a label for each atomic proposition
		for propNr, prop in enumerate(self.aprops):
			satisfied = ' | '.join(f'x={state}' for state in graph.states()
			                       if self.check_aprop(graph, propNr, state))

			print(f'label "{_translate_aprop(str(prop))}" = {satisfied if satisfied else "false"};',
	                      file=self.outfile)

		# Define a default reward specification for the number of steps
		print('\nrewards\n\t[] true: 1;\nendrewards', file=self.outfile)

	def make_reward(self, graph, name, evaluator):
		"""Generate a named reward specification for the given evaluator"""

		print(f'\nrewards "{name}"', file=self.outfile)

		for state in graph.states():
			value = evaluator(graph.getStateTerm(state))

			if value:
				print(f'\tx={state}: {value};', file=self.outfile)

		print('\nendrewards\n', file=self.outfile)
