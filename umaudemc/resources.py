#
# Import static resources (Maude files, HTML files...)
#
# These are included in the package tree and they should be loaded
# even if the package is distributed as a compressed archive. This is
# possible with importlib.resources.
#

import importlib.resources as pkg_resources

from . import data  # Import the data directory as a package

# Root for static resources
_DATA_ROOT = pkg_resources.files(data)


def get_resource_path(name):
	"""Get a temporary filename for a given named resource"""
	return pkg_resources.as_file(_DATA_ROOT / name)


def get_resource_content(name):
	"""Get the string content of a given named resource"""
	return (_DATA_ROOT / name).read_text()


def get_resource_binary(name):
	"""Get the string content of a given named resource"""
	return (_DATA_ROOT / name).read_bytes()


def get_templog():
	"""Get a temporary filename for the templog Maude file"""
	return get_resource_path('templog.maude')
