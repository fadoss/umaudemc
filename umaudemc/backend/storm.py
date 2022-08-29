#
# Storm probabilistic backend for umaudemc
#
# This file is wrapper to decide whether the library is available
# and use the command-line interface otherwise.
#

import importlib

if importlib.util.find_spec('stormpy'):
	from ._stormpy import *
else:
	from ._storm import *
