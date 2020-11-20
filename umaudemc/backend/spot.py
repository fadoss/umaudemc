#
# Spot backend for umaudemc
#
# This file is only a wrapper to detect whether the library
# is available. The actual backend is in _spot.py.
#

import importlib

if importlib.util.find_spec('spot'):
	from ._spot import *
else:
	class SpotBackend:
		def find(self):
			return False
