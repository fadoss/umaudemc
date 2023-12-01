#
# Spin backend for umaudemc
#

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from ..common import usermsgs
from ..formulae import collect_aprops
from ..wrappers import create_graph

# Translation to Spin's LTL syntax
# (the X operator is only available if Spin is built with -DNXT)
_translation = {
	'True'		: ('true',		0),
	'False'		: ('false',	 	0),
	'~_'		: ('! ({})',		1),
	'`[`]_'		: ('[] {}',		2),
	'<>_'		: ('<> {}',		2),
	'O_'		: ('X {}',		2),
	'_/\\_'		: ('{} && {}',		12),
	'_\\/_'		: ('{} || {}',		13),
	'_<->_'		: ('{} <-> {}',		14),
	'_->_'		: ('{} -> {}',		15),
	'_U_'		: ('({} U {})',		0),
	'_R_'		: ('({} V {})',		0),
}

# Regular expression for parsing NuSMV output
_STEP_REGEX = re.compile(br'\[s = (\d+)\]$')


def _preprocess_formula(formula):
	"""Preprocess the formula, simplifying some derived operators"""
	head, *rest = formula

	if head == 'Prop':
		return formula
	elif head == '_R_' and rest[0] == ['False']:
		return ['`[`]_', _preprocess_formula(rest[1])]
	elif head == '_U_' and rest[0] == ['True']:
		return ['<>_', _preprocess_formula(rest[1])]
	else:
		return [head] + [_preprocess_formula(arg) for arg in rest]


def _make_spin_formula(form, aprops, out_prio=20):
	"""Translate a formula to the Spin format"""
	head, *rest = form

	if head == 'Prop':
		result, prio = aprops[rest[0]], 0
	else:
		trans, prio = _translation[head]
		result = trans.format(*[_make_spin_formula(arg, aprops, prio) for arg in rest])

	return '(' + result + ')' if out_prio < prio else result


def make_spin_formula(form, aprops, ftype):
	"""Translate a formula to the Spin format"""
	return _make_spin_formula(_preprocess_formula(form), aprops)


class Spin:
	"""Spin backend connector"""

	def __init__(self):
		self.spin = None

	def find(self):
		"""Tries to find Spin"""
		if os.getenv('SPIN_PATH') is not None:
			spin_path = os.path.join(os.getenv('SPIN_PATH'), 'spin')

			if os.path.isfile(spin_path) and os.access(spin_path, os.X_OK):
				self.spin = spin_path
		# Look for it in the system path
		if not self.spin:
			self.spin = shutil.which('spin')

		return self.spin is not None

	def run(self, graph, form, ftype, aprops, extra_args=(), raw=False, timeout=None):
		"""Run Spin to check the following model-checking problem"""

		# If Spin has not been found, end with None
		if self.spin is None:
			return None, None

		# The model is written to a temporary file so that it can be read by Spin,
		# which apparently does not support direct input with pipes. Moreover, we
		# also reserve a temporary directory so that Spin writes its output there.

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpfile_name = os.path.join(tmpdir, 'model.pml')

			with open(tmpfile_name, 'w') as tmpfile:
				grapher = SpinGrapher(tmpfile, aprops=aprops)
				grapher.graph(graph)

				# Build the formula
				spin_formula = make_spin_formula(form, grapher.aprop_exprs, ftype)

				print(f'ltl p {{ {spin_formula} }}', file=tmpfile)

			# Record the time when the actual backend has been run
			start_time = time.perf_counter_ns()

			# Spin is called on the generated file
			try:
				status = subprocess.run((self.spin, '-run', 'model.pml', '-ltl', 'p')
				                        + tuple(extra_args), cwd=tmpdir,
				                        capture_output=not raw, timeout=timeout)

			except subprocess.TimeoutExpired:
				usermsgs.print_error(f'Spin execution timed out after {timeout} seconds.')
				return (None,) * 2

			if status.returncode != 0:
				usermsgs.print_error('An error was produced while running Spin:\n'
				                     + status.stdout[:-1].decode('utf-8'))
				os.remove(tmpfile.name)
				return None, None
			else:
				stats = {
					'states': graph.getNrStates(),
					'rewrites': grapher.getNrRewrites() + graph.getNrRewrites(),
					'backend_start_time': start_time
				}

				if graph.strategyControlled:
					stats['real_states'] = graph.getNrRealStates()

				# Spin writes a counterexample file when the property does not hold

				result = not any(line.startswith(b'pan: wrote model.pml.trail')
				                 for line in status.stdout.splitlines())

				if not result:
					# We have to call spin again with options -t -p to recover the
					# counterexample trace
					status = subprocess.run((self.spin, '-t', '-p', tmpfile_name),
					                        capture_output=True, cwd=tmpdir)

					if status.returncode != 0:
						usermsgs.print_warning('An error was produced while recovering '
						                       'the counterexample from Spin.')
						return result, stats

					# Parse the output of spin -t -p to recover the trace
					counterexample = ([0], [])
					current = 0

					for line in status.stdout.splitlines():
						if line.startswith(b'  <<<<<START OF CYCLE>>>>>'):
							current = 1

						elif line.startswith(b'spin: trail ends after'):
							break

						else:
							match = _STEP_REGEX.search(line)

							if match:
								counterexample[current].append(int(match.group(1)))

					stats['counterexample'] = counterexample

			return result, stats

	def check(self, graph=None, module=None, formula=None, logic=None, aprop_terms=None,
	          extra_args=(), get_graph=False, timeout=None, **kwargs):
		"""Solves a model-checking problem with Spin"""

		# Create the graph if not provided by the caller
		if graph is None:
			graph = create_graph(logic=logic, tableau=True, **kwargs)

		# Reparse the atomic propositions
		if aprop_terms is None:
			aprops = set()
			collect_aprops(formula, aprops)
			aprop_terms = [module.parseTerm(prop) for prop in aprops]

		holds, stats = self.run(graph, formula, logic, aprop_terms, extra_args, timeout=timeout)

		if holds is not None and get_graph:
			stats['graph'] = graph

		return holds, stats


class SpinGrapher:
	"""Graph writer for the Promela language"""

	def __init__(self, outfile=sys.stdout, aprops=(), slabel=None, elabel=None):
		"""
		Creates a grapher for Spin modules.

		:param outfile: Output file where the module will be written.
		:type outfile: file-like object
		:param aprops: Terms for each atomic proposition to be listed in the module
		:type aprops: list of maude.Term
		:param slabel: State label printing function
		:type slabel: function that receives a graph and state number
		:param elabel: Edge label printing function (will be ignored)
		:type elabel: any
		"""
		self.visited = set()
		self.aprops = aprops
		self.outfile = outfile
		self.slabel = slabel if slabel is not None else lambda *args: ""

		self.aprop_exprs = {}

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

	def graph(self, graph, bound=-1):
		"""Generate the graph with a given bound limit"""
		self.explore(graph, 0, bound)

		# Find the satisfaction (|=) symbol and the true constant to be used
		# when testing atomic propositions.

		module = graph.getStateTerm(0).symbol().getModule()
		boolkind = module.findSort('Bool').kind()

		self.satisfies = module.findSymbol('_|=_',
		                                   (module.findSort('State').kind(), module.findSort('Prop').kind()),
		                                   boolkind)

		self.true_term = module.parseTerm('true', boolkind)

		# Declares a single variable 'state' of integer type that matches with
		# the internal state numbers of the graph.

		print(f'int s;', file=self.outfile)

		# Atomic propositions are given as explicit expression within formulae,
		# we calculate them here and write them for debugging

		for prop in range(len(self.aprops)):
			expr = ' || '.join(f's == {state}' for state in self.visited
			                   if self.check_aprop(graph, prop, state))

			expr = f'({expr})' if expr else 'false'

			self.aprop_exprs[str(self.aprops[prop])] = expr

			print(f'/* {self.aprops[prop]} = {expr} */', file=self.outfile)

		# The initial state is always zero

		print('init { s = 0 }', file=self.outfile)

		# The transition relation is specified using a repetition statement with guards
		# in the single process of the specification

		if len(self.visited) == 1:
			return

		print('active proctype P() {\n  do', file=self.outfile)

		for state in self.visited:
			# Write the label for each state before its entry (unless empty)
			comment = self.slabel(graph, state)
			if str(comment):
				print(f'  /* {comment} */', file=self.outfile)

			# Every transition from the current state is written as an atomic statement
			# that assigns the state variable to the next state index

			for next_state in graph.getNextStates(state):
				print(f'  :: atomic {{ s == {state} -> s = {next_state} }}', file=self.outfile)

		print('  od\n}', file=self.outfile)

	def explore(self, graph, stateNr, bound=-1):
		"""Explore the graph up to an optional given depth, adding the nodes to the visited set"""
		# Depth-first search with a stack
		pending = [(stateNr, bound)]
		self.visited.add(stateNr)

		if bound == 0:
			return

		while pending:
			state, limit = pending.pop()

			for next_state in graph.getNextStates(state):
				# Append the state to be visited if not already done
				if next_state not in self.visited:
					self.visited.add(next_state)

					if bound == -1:
						pending.append((next_state, -1))
					elif limit > 1:
						pending.append((next_state, limit - 1))
