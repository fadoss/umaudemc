#!/usr/bin/env python3
#
# Test driver for umaudemc
#

import difflib
import itertools
import os
import subprocess
import stat
import sys
import tempfile

from pathlib import Path


class TestDriver:
	"""Test driver for umaudemc by text output comparison"""

	POSIX_ADAPTER = f'#!/bin/sh\nexec {sys.executable} -m umaudemc "$@"\n'
	WIN32_ADAPTER = f'@echo off\n{sys.executable} -m umaudemc %*\n'

	DIFF_HEADER = '''<!DOCTYPE html>
		<html>
			<head>
				<meta charset="utf-8">
				<title>Test differences</title>
				<style type="text/css">
					table.diff {font-family: monospace; border:medium;}
					.diff_header {background-color:#e0e0e0}
					td.diff_header {text-align:right}
					.diff_next {background-color:#c0c0c0}
					.diff_add {background-color:#aaffaa}
					.diff_chg {background-color:#ffff77}
					.diff_sub {background-color:#ffaaaa}
				</style>
			</head>
			<body>'''

	def __init__(self, write_diff=False):
		# Base directory of the umaudemc package
		self.basedir = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
		# Adapted environment for running the tests
		self.environ = os.environ.copy()
		self.environ['PYTHONPATH'] = self.basedir

		# In order to use umaudemc safely as a command in a portable way, we create
		# a temporary directory with a umaudemc script and add it to the path
		self.extra_path = tempfile.TemporaryDirectory()
		self.environ['PATH'] = f'{self.extra_path.name}{os.pathsep}{self.environ["PATH"]}'

		if os.name == 'nt':
			adapter_name, adapter = 'umaudemc.bat', self.WIN32_ADAPTER
		else:
			adapter_name, adapter = 'umaudemc', self.POSIX_ADAPTER

		adapter_name = Path(self.extra_path.name) / adapter_name
		adapter_name.write_text(adapter)
		adapter_name.chmod(stat.S_IRWXU)

		# Detect the available backends
		sys.path = [self.basedir] + sys.path

		from umaudemc.backends import get_backends, DEFAULT_BACKENDS
		self.available, _ = get_backends(DEFAULT_BACKENDS)
		self.available = {backend[0] for backend in self.available}

		from umaudemc.backend.prism import PRISMBackend
		from umaudemc.backend.storm import StormBackend

		if PRISMBackend().find():
			self.available.add('prism')

		if StormBackend().find():
			self.available.add('storm')

		# Check whether using colored output or not
		if os.isatty(1):
			self.ok = '\x1b[32mok\x1b[0m'
			self.fail = '\x1b[31;1merror\x1b[0m'
		else:
			self.ok = 'ok'
			self.fail = 'error'

		# Diff generation
		self.differ = difflib.HtmlDiff() if write_diff else None
		self.diff_file = None

	def finish(self):
		"""Finish what should be finished"""

		if self.diff_file is not None:
			self.diff_file.write('</body></html>')
			self.diff_file.close()
			self.diff_file = None

	def write_diff(self, testname, expected, output):
		"""Write the differences to a file"""

		if self.diff_file is None:
			self.diff_file = open('test_diff.htm', 'w')
			self.diff_file.write(self.DIFF_HEADER)

		expected_lines = expected.decode('utf-8').split('\n')
		output_lines = output.decode('utf-8').split('\n')

		table = self.differ.make_table(expected_lines, output_lines,
		                               fromdesc='Expected', todesc='Obtained',
		                               context=True)

		self.diff_file.write(f'<h1>{testname}</h1>{table}')

	def compare_output(self, testname, expected_file, output):
		"""Compare the expected and the actual output"""

		expected = Path(expected_file).read_bytes()

		the_same = expected == output

		if not the_same and self.differ:
			self.write_diff(testname, expected, output)

		return the_same

	@staticmethod
	def get_annotation(test_file):
		"""Get the annotation in the header of the test case file"""

		annot = None

		# The annotation must be in the first two lines of the file
		with open(test_file) as tf:
			for line in itertools.islice(tf, 2):
				if line.startswith('#+'):
					annot = line
					break

		if not annot:
			return {}

		# It is a dictionary of comma-separated entries where they key
		# and the value are separated by a colon. The value is parsed
		# as a space-separated list.
		annot = [l.strip().split(':') for l in line[2:].split(',')]
		annot = {name: value.strip().split(' ') for name, value in annot}

		return annot

	def set_backend(self, backend):
		"""Select a specific backend"""

		self.environ['UMAUDEMC_BACKEND'] = backend
		self.environ['UMAUDEMC_PBACKEND'] = backend

	def reset_backend(self):
		"""Reset the backend selection"""

		self.environ['UMAUDEMC_BACKEND'] = self.environ.get('UMAUDEMC_BACKEND')
		self.environ['UMAUDEMC_PBACKEND'] = self.environ.get('UMAUDEMC_PBACKEND')

	def do_run_test(self, dirname, filename, outname):
		"""Run the test using the system interpreter"""

		status = subprocess.run(['cmd' if os.name == 'nt' else 'sh', filename], cwd=dirname,
		                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
		                        env=self.environ)

		out_file = os.path.join(dirname, outname + '.out')
		expected_file = os.path.join(dirname, outname + '.expected')

		the_same = False

		if not os.path.exists(expected_file):
			print('first time')

		else:
			the_same = self.compare_output(outname, expected_file, status.stdout)
			print(self.ok if the_same else self.fail)

			if the_same and os.path.exists(out_file):
				os.remove(out_file)

		if not the_same:
			Path(out_file).write_bytes(status.stdout)

	def run_test(self, dirname, filename):
		"""Run a single test taking its header into account"""

		testname, ext = os.path.splitext(filename)

		print(f'  {testname:20} ', end='', flush=True)

		# We consider two annotations in the test header:
		# - 'for' specifies a list of backend with which to run the test
		# - 'req' specifies a backend that is required to run the test
		annots = self.get_annotation(os.path.join(dirname, filename))

		variants = annots.get('for', ())

		# Check whether the required backends are available
		if any(backend not in self.available for backend in annots.get('req', ())):
			print('unavailable')

		# The test is run with the default backend if 'for' is absent
		elif len(variants) == 0:
			self.do_run_test(dirname, filename, testname)

		elif len(variants) == 1:
			self.set_backend(variants[0])
			self.do_run_test(dirname, filename, testname)
			self.reset_backend()

		else:
			print()
			for backend in variants:
				print(f'    {backend:18} ', end='', flush=True)

				if backend not in self.available:
					print('unavailable')
				else:
					self.set_backend(backend)
					self.do_run_test(dirname, filename, f'{testname}_{backend}')
					self.reset_backend()

	def run_dir(self, dirname):
		"""Run all tests in a directory, recursively"""

		for path, dirs, files in os.walk(dirname):
			# Avoid printing the directory name if there is no test in it
			if not any(f.endswith('.test') for f in files):
				continue

			print(os.path.basename(path))

			for file in files:
				if file.endswith('.test'):
					self.run_test(path, file)

	def run_tests(self, tests):
		"""Run the given list of tests"""

		# If no test is given, we consider the current working directory
		if not tests:
			tests = [os.curdir]

		for test in tests:
			if os.path.isdir(test):
				self.run_dir(test)
			else:
				self.run_test(os.path.dirname(test), os.path.basename(test))


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Test driver for umaudemc')
	parser.add_argument('test', nargs='*', help='List of test to be executed (nothing for all)')
	parser.add_argument('--no-diff', help='Do not generate a diff file', dest='write_diff', action='store_false')

	args = parser.parse_args()

	driver = TestDriver(write_diff=args.write_diff)
	driver.run_tests(args.test)
	driver.finish()
