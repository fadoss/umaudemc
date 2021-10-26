#
# Storm probabilistic backend for umaudemc
#

import os
import re
import shutil
import subprocess

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
			storm_path = os.path.join(os.getenv('STORM_PATH'), 'storm')

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
			                     + status.stdout[:-1].decode('utf-8'))
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
