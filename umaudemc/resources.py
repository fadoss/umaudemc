#
# Import static resources (Maude files, HTML files...)
#
# These are included in the package tree and they should be loaded
# even if the package is distributed as a compressed archive. This is
# possible with importlib.resources in 3.7 and its backport
# importlib_resources in previous versions.
#

import sys
import importlib.resources as pkg_resources

from . import data  # Import the data directory as a package

# To avoid deprecation warnings in Python 3.11 and above
if sys.version_info >= (3, 9):
	
	def get_resource_path(name):
		"""Get a temporary filename for a given named resource"""
		return pkg_resources.as_file(pkg_resources.files(data) / name)

	def get_resource_content(name):
		"""Get the string content of a given named resource"""
		return (pkg_resources.files(data) / name).read_text()

	def get_resource_binary(name):
		"""Get the string content of a given named resource"""
		return (pkg_resources.files(data) / name).read_bytes()

# For compatibility with Python 3.8
else:
	def get_resource_path(name):
		"""Get a temporary filename for a given named resource"""
		return pkg_resources.path(data, name)


	def get_resource_content(name):
		"""Get the string content of a given named resource"""
		return pkg_resources.read_text(data, name)


	def get_resource_binary(name):
		"""Get the string content of a given named resource"""
		return pkg_resources.read_binary(data, name)

def get_templog():
	"""Get a temporary filename for the templog Maude file"""
	return get_resource_path('templog.maude')
