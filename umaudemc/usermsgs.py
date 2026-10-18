#
# Generic user message functions
#

import sys

from .terminal import terminal as tmn


def print_warning(msg):
	print(f'{tmn.yellow}{msg}{tmn.reset}', file=sys.stderr)


def print_error(msg):
	print(f'{tmn.bright_red}{msg}{tmn.reset}', file=sys.stderr)


def print_info(msg):
	print(f'{tmn.cyan}{msg}{tmn.reset}', file=sys.stderr)


def print_error_loc(unit, line, column, msg):
	"""Print an error with a location"""

	print(f'{tmn.bold}{unit}:{line}:{column}: {tmn.red}error:{tmn.reset} {msg}', file=sys.stderr)


def print_info_loc(unit, line, column, msg):
	"""Print an error with a location"""

	print(f'{tmn.bold}{unit}:{line}:{column}: {tmn.cyan}info:{tmn.reset} {msg}', file=sys.stderr)


def print_error_file(msg, unit):
	"""Print an error with a location"""

	print(f'{tmn.bold}{unit}: {tmn.red}error:{tmn.reset} {msg}', file=sys.stderr)


def print_warning_loc(unit, line, column, msg):
	"""Print a warning with a location"""

	print(f'{tmn.bold}{unit}:{line}:{column}: {tmn.yellow}warning:{tmn.reset} {msg}', file=sys.stderr)
