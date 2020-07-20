#
# PyModelChecking backend for umaudemc
#
# This file is only a wrapper to detect whether the library
# is available. The actual backend is in _pymc.py.
#

import importlib

if importlib.util.find_spec('pyModelChecking'):
	from ._pymc import *
else:
	class PyModelChecking:
		def find(self):
			return False
