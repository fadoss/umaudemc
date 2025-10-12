#
# Command for a worker of the statistical model checker
#

import json
import multiprocessing as mp
import os
import random
import socket
import sys
import tarfile
import tempfile
from array import array

from ..common import maude, parse_initial_data, usermsgs
from ..quatex import parse_quatex
from ..simulators import get_simulator
from ..statistical import run, QueryData, make_parameter_dicts

# Python-version-dependent option for safer TAR extraction
EXTRACT_OPTIONS = {'filter': 'data'} if sys.version_info >= (3, 12) else {}


def read_json(sock):
	"""Read a JSON value from a socket"""

	# Read a 32 bit integer for the value length in bytes
	if not (data := sock.recv(4)):
		return None

	length = int.from_bytes(data, 'big')

	# Read the JSON to a byte string
	data = b''

	while len(data) < length:
		data += sock.recv(length)

	return json.loads(data.decode())


class Dummy:
	"""Dummy class to be used as a namespace"""

	def __init__(self, values):
		self.__dict__ = values


class Worker:
	"""Worker for simulations"""

	def __init__(self):
		self.program = None
		self.simulator = None
		self.block = 100

	def setup(self, tmp_dir, args):
		"""Setup the execution environment"""

		# Copy parameters
		self.block = args.block

		maude.setRandomSeed(args.seed)
		random.seed(args.seed)

		# Do the same as the scheck command without checks
		args.file = os.path.join(tmp_dir, args.file)

		if not (data := parse_initial_data(args)):
			return False

		with open(os.path.join(tmp_dir, args.query)) as quatex_file:
			self.program, _ = parse_quatex(quatex_file, filename=args.query,
			                               legacy=args.assign == 'pmaude',
						       constants=args.constants)

		if not self.program:
			return False

		# Get the simulator for the given assignment method
		self.simulator = get_simulator(args.assign, data)

		if not self.simulator:
			return False

		return True

	def run(self, conn):
		"""Run the simulation until it is finished"""

		program = self.program
		simulator = self.simulator
		block = self.block

		# Query data
		# (delta, its second argument, does not matter because
		# convergence is not evaluated by the worker)
		qdata = [QueryData(k, 1.0, idict)
		         for k, qinfo in enumerate(program.query_locations)
		         for idict in make_parameter_dicts(qinfo[3])]

		sums = array('d', [0.0] * len(qdata))
		sum_sq = array('d', [0.0] * len(qdata))
		counts = array('i', [0] * len(qdata))

		while True:

			for _ in range(block):
				# Run the simulation and compute all queries at once
				values = run(program, qdata, simulator)

				for k, value in enumerate(values):
					if value is not None:
						sums[k] += value
						sum_sq[k] += value * value
						counts[k] += 1

			conn.send(b'b' + sums.tobytes() + sum_sq.tobytes() + counts.tobytes())

			# Check whether to continue
			answer = conn.recv(1)

			if answer == b's':
				print('Done')
				return

			elif answer != b'c':
				usermsgs.print_error(f'Unknown command {answer.decode()}. Stopping.')
				return

			for k in range(len(qdata)):
				sums[k] = 0
				sum_sq[k] = 0
				counts[k] = 0


def handle_request(message, conn, addr, keep_file):
	"""Handle a request in a separate process"""

	command = Dummy(message)

	with tempfile.TemporaryDirectory(delete=not keep_file) as tmp_dir:
		# Print the temporary working directory for debugging purposes
		if keep_file:
			print('Temporary directory:', tmp_dir)

		# Recover the required files
		with conn.makefile('rb', buffering=0) as fobj:
			with tarfile.open(mode='r|*', fileobj=fobj) as tarf:
				tarf.extractall(tmp_dir, **EXTRACT_OPTIONS)

		# Setup a worker object
		worker = Worker()

		if not worker.setup(tmp_dir, command):
			usermsgs.print_error(f'{addr}: bad request from scheck')
			conn.send(b'e')
			return

		# Send confirmation
		conn.send(b'o')

		# Wait for start signal
		conn.recv(1)

		worker.run(conn)


def sworker(args):
	"""Worker for distributed statistical model checking"""

	# Parse the listing address
	try:
		with socket.create_server((args.address, args.port), backlog=1) as sock:
			while True:
				print(f'👂 Listening on {args.address}:{args.port}...')
				conn, addr = sock.accept()

				usermsgs.print_info(f'Accepted connection from {":".join(map(str, addr))}.')

				while True:
					# Read the initiation message
					if (message := read_json(conn)) is None:
						break

					# A separate process to cleanup Maude state
					process = mp.Process(target=handle_request, args=(message, conn, addr, args.keep_file))
					process.start()
					process.join()

	except KeyboardInterrupt:
		print('Server closed by the user.')

	return 0
