#
# Generic user message functions
#

from .terminal import terminal as tmn


def print_warning(msg):
	print(f'{tmn.yellow}{msg}{tmn.reset}')


def print_error(msg):
	print(f'{tmn.bright_red}{msg}{tmn.reset}')


def print_info(msg):
	print(msg)


def print_error_loc(unit, line, column, msg):
	"""Print an error with a location"""

	print(f'{tmn.bold}{unit}:{line}:{column}: {tmn.red}error:{tmn.reset} {msg}')


def print_warning_loc(unit, line, column, msg):
	"""Print a warning with a location"""

	print(f'{tmn.bold}{unit}:{line}:{column}: {tmn.yellow}warning:{tmn.reset} {msg}')
