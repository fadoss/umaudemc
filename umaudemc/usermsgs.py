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
