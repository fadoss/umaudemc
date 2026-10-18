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
		self.dump = None

	def setup(self, tmp_dir, args):
		"""Setup the execution environment"""

		# Copy parameters
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

		# Setup the dump
		if args.dump:
			self.dump = open(os.path.join(tmp_dir, 'dump.log'), 'w')

		return True

	def run(self, conn):
		"""Run the simulation until it is finished"""

		# Wait for start signal and read block size
		conn.recv(1)
		block = int.from_bytes(conn.recv(4), signed=False)

		program = self.program
		simulator = self.simulator

		# Query data
		# (its second and last arguments do not matter because
		# convergence is not evaluated by the worker)
		qdata = [QueryData(k, 1.0, idict, None)
		         for k, qinfo in enumerate(program.queries)
		         for idict, _ in make_parameter_dicts(qinfo, 1.0)]

		# Compact arrays are used so that they can be sent through the socket
		sums = array('d', [0.0] * len(qdata))
		sum_sq = array('d', [0.0] * len(qdata))
		counts = array('i', [0] * len(qdata))

		converged = array('i')  # to store indices of converged queries

		while True:

			for _ in range(block):
				# Run the simulation and compute all queries at once
				values = run(program, qdata, simulator)

				# Dump evaluations if required
				if self.dump:
					print(*values, file=self.dump)

				for k, value in enumerate(values):
					if value is not None:
						sums[k] += value
						sum_sq[k] += value * value
						counts[k] += 1

			conn.send(b'b' + sums.tobytes() + sum_sq.tobytes() + counts.tobytes())

			# Check whether to continue
			answer = conn.recv(1)

			# Stop command
			if answer == b's':
				print('Done')

				# Send the dump if requested
				if self.dump:
					self.dump.close()
					conn.send(b'd')
					with open(self.dump.name, 'rb') as dfile:
						conn.sendfile(dfile)
					conn.shutdown(socket.SHUT_WR)

				return

			# Partial convergence
			elif answer == b'p':
				mcount = int.from_bytes(conn.recv(4), 'big')
				# Safety check
				if mcount <= len(qdata):
					converged.frombytes(conn.recv(4 * mcount))
					# Set the converged queries to avoid recomputing them
					for index in converged:
						qdata[index].converged = True
					del converged[:]

			elif answer != b'c':
				usermsgs.print_error(f'Unknown command {answer.decode()}. Stopping.')
				if self.dump:
					self.dump.close()
					self.dump = None
				return

			# Receive the new block size
			block = int.from_bytes(conn.recv(4), signed=False)

			# Reset the accumulators for the next round
			for k in range(len(qdata)):
				sums[k] = 0
				sum_sq[k] = 0
				counts[k] = 0


def handle_request(message, conn, addr, keep_file):
	"""Handle a request in a separate process"""

	command = Dummy(message)

	# For compatibility with versions before 3.12
	tmpdir_opts = {'delete': not keep_file} if sys.version_info >= (3, 12) else {}

	with tempfile.TemporaryDirectory(**tmpdir_opts) as tmp_dir:
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

		worker.run(conn)


def sworker(args):
	"""Worker for distributed statistical model checking"""

	# Parse the listing address
	try:
		with socket.create_server((args.address, args.port), backlog=1) as sock:
			while True:
				print(f'👂 Listening on {args.address}:{args.port}...')
				conn, addr = sock.accept()
				addr_str = ':'.join(map(str, addr))

				usermsgs.print_info(f'Accepted connection from {addr_str}.')

				while True:
					# Read the initiation message
					if (message := read_json(conn)) is None:
						break

					# A separate process to cleanup Maude state
					process = mp.Process(target=handle_request, args=(message, conn, addr_str, args.keep_file))
					process.start()
					process.join()

					# Check whether process has failed
					if process.exitcode != 0:
						usermsgs.print_error(f'{addr_str}: worker process failed with exit code {process.exitcode}')

						try: conn.send(b'e')
						except ConnectionError: pass

	except KeyboardInterrupt:
		print('Server closed by the user.')

	return 0
