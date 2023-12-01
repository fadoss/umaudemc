#
# NuSMV backend for umaudemc
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

# Translation to NuSMV's CTL syntax
# (the priorities may not be completely accurate)
_ctl_translation = {
	'True'		: ('TRUE',		0),
	'False'		: ('FALSE',	 	0),
	'~_'		: ('! ({})',		1),
	'`[`]_'		: ('G {}',		2),
	'<>_'		: ('F {}',		2),
	'O_'		: ('X {}',		2),
	'A_'		: ('A{}',		3),
	'E_'		: ('E{}',		3),
	'_/\\_'		: ('{} & {}',		12),
	'_\\/_'		: ('{} | {}',		13),
	'_<->_'		: ('{} <-> {}',		14),
	'_->_'		: ('{} -> {}',		15),
	'_U_'		: (' [ {} U {} ]',	0)
}

# Translation to LTL syntax (a copy of CTL where certain entries will
# not be used)
_ltl_translation = _ctl_translation.copy()
_ltl_translation['_U_'] = ('{} U {}', 12)

# Regular expression for parsing NuSMV output
_RESULT_REGEX = re.compile(b'-- specification .* is (false|true)')
_COUNTER_REGEX = re.compile(r'^(\s+)state = (\d+)$')
_LOOP_REGEX = re.compile(r'^\s+-- Loop starts here$')


def _preprocess_formula(formula):
	"""Preprocess the formula, translating the unsupported operators"""
	head, *rest = formula

	if head == 'Prop':
		return formula
	elif head == '_R_' and rest[0] == ['False']:
		return ['`[`]_', _preprocess_formula(rest[1])]
	elif head == '_R_':
		left  = ['~_', _preprocess_formula(rest[1])]
		right = ['~_', _preprocess_formula(rest[0])]
		return ['~_', ['_U_', left, right]]
	elif head == '_U_' and rest[0] == ['True']:
		return ['<>_', _preprocess_formula(rest[1])]
	else:
		return [head] + [_preprocess_formula(arg) for arg in rest]


def _make_smv_formula(form, aprops, translation, out_prio=20):
	"""Translate a formula to the NuSMV format"""
	head, *rest = form

	if head == 'Prop':
		result, prio = 'state in p{}'.format(aprops.index(rest[0])), 0
	else:
		trans, prio = translation[head]
		result = trans.format(*[_make_smv_formula(arg, aprops, translation, prio) for arg in rest])

	return '(' + result + ')' if out_prio < prio else result


def make_smv_formula(form, aprops, ftype):
	"""Translate a formula to the NuSMV format"""
	return _make_smv_formula(_preprocess_formula(form),
	                         aprops,
	                         _ctl_translation if ftype == 'CTL' else _ltl_translation)


class NuSMV:
	"""NuSMV backend connector"""

	def __init__(self):
		self.nusmv = None

	def find(self):
		"""Tries to find NuSMV"""
		if os.getenv('NUSMV_PATH') is not None:
			nusmv_path = os.path.join(os.getenv('NUSMV_PATH'), 'NuSMV')

			if os.path.isfile(nusmv_path) and os.access(nusmv_path, os.X_OK):
				self.nusmv = nusmv_path
		# Look for it in the system path
		if not self.nusmv:
			self.nusmv = shutil.which('NuSMV')

		return self.nusmv is not None

	def run(self, graph, form, ftype, aprops, extra_args=(), raw=False, timeout=None):
		"""Run NuSMV to check the following model-checking problem"""

		# If NuSMV has not been found, end with None
		if self.nusmv is None:
			return None, None

		# Create the formula
		smv_formula = make_smv_formula(form, list(map(str, aprops)), ftype)

		# The model is written to a temporary file so that it can be read by NuSMV,
		# which apparently does not support direct input with pipes.
		#
		# Since in Windows a file cannot be opened twice by default, we are forced
		# to pass delete=False to NamedTemporaryFile and remove it manually.

		with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
			grapher = NuSMVGrapher(tmpfile, aprops=aprops, stutter_ext=not graph.strategyControlled)

			grapher.graph(graph)
			print(('SPEC ' if ftype == 'CTL' else 'LTLSPEC ') + smv_formula, file=tmpfile)
			tmpfile.flush()

		# Record the time when the actual backend has been run
		start_time = time.perf_counter_ns()

		# NuSMV is called on the generated file
		try:
			status = subprocess.run((self.nusmv, tmpfile.name) + tuple(extra_args),
			                        capture_output=not raw, timeout=timeout)

		except subprocess.TimeoutExpired:
			usermsgs.print_error(f'NuSMV execution timed out after {timeout} seconds.')
			return (None,) * 2

		if status.returncode != 0:
			usermsgs.print_error('An error was produced while running NuSMV:\n'
			                     + status.stderr[:-1].decode('utf-8'))
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

			# Parse the NuSMV output to obtain the binary result of the check and
			# the counterexample in such case.

			result = None
			counterexample = [], []
			counter_part = 0

			for line in status.stdout.splitlines():
				if result is None:
					match = _RESULT_REGEX.match(line)
					if match:
						result = match.group(1) == b'true'
				else:
					# Read the counterexample storing the state number into the
					# lead-in and cycle list counter_part index, which is stepped
					# when the loop start message is found.

					line = line.decode('utf-8')
					match = _COUNTER_REGEX.match(line)
					if match:
						counterexample[counter_part].append(int(match.group(2)))

					elif _LOOP_REGEX.match(line) and counter_part == 0:
						# When the loop is a self loop of the last state, NuSMV
						# prints "loop starts here" twice, we ignore the second.
						counter_part = 1

			# Add the counterexample to the statistics
			if counterexample != ([], []):
				# NuSMV includes the start of the loop twice, unlike Maude
				if len(counterexample[1]) > 1:
					counterexample[1].pop()

				stats['counterexample'] = counterexample

			os.remove(tmpfile.name)
			return result, stats

	def check(self, graph=None, module=None, formula=None, logic=None, aprop_terms=None,
	          extra_args=(), get_graph=False, timeout=None, **kwargs):
		"""Solves a model-checking problem with NuSMV"""

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


class NuSMVGrapher:
	"""Graph writer in NuSMV input format"""

	def __init__(self, outfile=sys.stdout, aprops=(), slabel=None, elabel=None, stutter_ext=False):
		"""
		Creates a grapher for NuSMV modules.

		:param outfile: Output file where the module will be written.
		:type outfile: file-like object
		:param aprops: Terms for each atomic proposition to be listed in the module
		:type aprops: list of maude.Term
		:param slabel: State label printing function
		:type slabel: function that receives a graph and state number
		:param elabel: Edge label printing function (will be ignored)
		:type elabel: any
		:param stutter_ext: Whether deadlock states should be applied the stutter extension
		:type stutter_ext: bool
		"""
		self.visited = set()
		self.aprops = aprops
		self.outfile = outfile
		self.slabel = slabel if slabel is not None else lambda *args: ""
		self.stutter_ext = stutter_ext

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

		self.true_term = module.findSymbol('true', (), boolkind).makeTerm(())

		# Start the NuSMV module and declares a single variable 'state' of bounded
		# integer type that matches with the internal state numbers of the graph.

		max_state = max(self.visited)

		print(f'MODULE main\nVAR\n  state: 0..{max_state};', file=self.outfile)

		# Atomic propositions are defined as numbered set constants of the form pn.
		# The elements of the set are the states in which the property is satisfied.
		# Since empty sets are not supported, its role is safely played by {-1}.

		print('DEFINE', file=self.outfile)
		for prop in range(len(self.aprops)):
			print(f'    -- {self.aprops[prop]}', file=self.outfile)
			satisfied = {state for state in self.visited if self.check_aprop(graph, prop, state)}
			if satisfied == set():
				satisfied = {-1}
			print(f'    p{prop} := {satisfied};', file=self.outfile)

		# The initial state is always zero

		print('INIT\n  state in 0;', file=self.outfile)

		# The transition relation is specified with a case construct, which assigns
		# to each possible state the set of values next(state) may take.

		print('TRANS\n  next(state) in case', file=self.outfile)

		for state in range(max_state + 1):
			# Write the label for each state before its entry (unless empty)
			comment = self.slabel(graph, state)
			if str(comment):
				print(f'    -- {comment}', file=self.outfile)

			# NuSMV requires the case distinction to be exhaustive in the integer
			# range, so intermediate states that do not really belong to the graph
			# (e.g. purged failed states...) must also appear.

			if state in self.visited:
				successors = {next_state for next_state in graph.getNextStates(state)}
				if successors == set():
					successors = {state} if self.stutter_ext else {-1}
			else:
				successors = {-1}

			print(f'    state = {state} : {successors};', file=self.outfile)

		print('  esac;', file=self.outfile)

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
