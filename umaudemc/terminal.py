#
# Terminal helper functions
#

import sys


class ANSITermHelper:
	"""Class with attributes for pretty-printing in an ANSI terminal"""

	# Reset
	reset   = '\x1b[0m'
	# Features
	bold    = '\x1b[1m'
	italic  = '\x1b[3m'
	# Colors
	red     = '\x1b[31m'
	green   = '\x1b[32m'
	yellow  = '\x1b[33m'
	blue    = '\x1b[34m'
	magenta = '\x1b[35m'
	cyan    = '\x1b[36m'

	bright_red = '\x1b[91m'
	bright_green = '\x1b[92m'

	# Commands
	clean_line = '\x1b[2K'


class DummyTermHelper:
	"""Dummy class with the same attributes as ANSITermHelper"""

	# Reset
	reset   = ''
	# Features
	bold    = ''
	italic  = ''
	# Colors
	red     = ''
	green   = ''
	yellow  = ''
	blue    = ''
	magenta = ''
	cyan    = ''

	bright_red = ''
	bright_green = ''

	# Commands
	clean_line = ''


def get_term_helper(fo, want_color=True):
	"""Get the appropriate terminal helper for the given file object"""

	return ANSITermHelper if want_color and fo.isatty() else DummyTermHelper


# Terminal helper for the standard output
terminal = get_term_helper(sys.stdout)
