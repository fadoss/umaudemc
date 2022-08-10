#
# Storm probabilistic backend for umaudemc
#

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from . import prism
from .. import probabilistic as pbs
from ..common import usermsgs

# Result line printed by Storm
_RESULT_REGEX = re.compile(b'^Result \\(for initial states\\): ([\\d.E-]+|inf|false|true)')


class StormBackend(prism.PRISMBackend):
	"""Storm backend"""

	def find(self):
		"""Tries to find Storm"""

		if os.getenv('STORM_PATH') is not None:
			storm_path = os.getenv('STORM_PATH')

			if not os.path.isfile(storm_path):
				storm_path = os.path.join(storm_path, 'storm')

			if os.path.isfile(storm_path) and os.access(storm_path, os.X_OK):
				self.command = storm_path

		# Look for it in the system path
		if not self.command:
			self.command = shutil.which('storm')

		return self.command is not None

	def run_command(self, filename, formula, extra_args, timeout, raw):
		"""Run Storm and parse the result"""

		try:
			status = subprocess.run([self.command, '--prism', filename, '--prop', formula] + list(extra_args),
			                        capture_output=not raw, timeout=timeout)

		except subprocess.TimeoutExpired:
			# Storm can handle timeouts by itself
			usermsgs.print_error(f'Storm execution timed out after {timeout} seconds.')
			return None

		if status.returncode != 0:
			usermsgs.print_error('An error was produced while running Storm:\n'
			                     + status.stdout[:-1].decode('utf-8')
			                     + status.stderr[:-1].decode('utf-8'))
			return None

		# Parse the Storm output to obtain the result
		result = None

		for line in status.stdout.splitlines():
			match = _RESULT_REGEX.match(line)
			# The result can be true, false, or a probability
			if match:
				token = match.group(1)

				if token == b'true':
					result = pbs.QuantitativeResult.make_boolean(True)
				elif token == b'false':
					result = pbs.QuantitativeResult.make_boolean(False)
				else:
					# If this result is not the first one, we consider it is
					# range (for the MDP case). Raw formulae may produce more
					# results, but this is not supported.
					if result is None:
						result = pbs.QuantitativeResult.make_number(float(token))
					else:
						result = pbs.QuantitativeResult.make_range(result, float(token))

			# Errors are printed by Storm on the standard output
			if line.startswith(b'ERROR '):
				usermsgs.print_error('Error: ' + line[6:].decode('utf-8'))

		return result

	def state_analysis(self, step, graph=None, extra_args=(), timeout=None, raw=False):
		"""Steady and transient state analysis"""

		if self.command is None:
			return None, None

		# Transient probabilities are not calculated by Storm, as far as we know
		if step is not None:
			usermsgs.print_error('Transient probabilities are not supported by Storm. Use PRISM instead.')
			return None, None

		# The model is written to a temporary file, like in run.

		with tempfile.TemporaryDirectory() as tmpdir:
			model_file = os.path.join(tmpdir, 'model.pm')
			export_file = os.path.join(tmpdir, 'export.json')

			with open(model_file, 'w') as pm:
				grapher = prism.PRISMGenerator(pm, aprops=set())

				# Output the DTMC or MDP for the model
				grapher.graph(graph)

			# Storm invocation for steady-state probabilities
			cmd_args = [self.command, '--prism', model_file, '--buildstateval',
			            '--steadystate', '--exportresult', export_file]

			# Record the time when the actual backend has run for statistics
			start_time = time.perf_counter_ns()

			try:
				status = subprocess.run(cmd_args + list(extra_args),
				                        stdout=None if raw else subprocess.PIPE,
				                        stderr=subprocess.STDOUT, timeout=timeout)

			except subprocess.TimeoutExpired:
				usermsgs.print_error(f'Storm execution timed out after {timeout} seconds.')
				return None

			# If something has go wrong...
			if status.returncode != 0:
				usermsgs.print_error('An error was produced while running Storm:\n'
				                     + status.stdout[:-1].decode('utf-8')
				                     + status.stderr[:-1].decode('utf-8'))
				return None, None

			elif b'WARN' in status.stdout:
				for line in status.stdout.split(b'\n'):
					if b'WARN' in line:
						line = line.strip().decode('utf-8')
						usermsgs.print_warning(line.replace('WARN', 'Warning from Storm'))

			# The output file is a JSON with the probabilities
			with open(export_file) as ef:
				ssout = json.load(ef)
				result = [0.0] * len(ssout)
				for entry in ssout:
					result[entry['s']['x']] = entry['v']

		return result, self.make_statistics(grapher, graph, start_time)
