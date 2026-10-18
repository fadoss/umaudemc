#
# Distributed model checking
#

import ctypes
import io
import json
import os
import random
import re
import selectors
import socket
import sys
import tarfile
from array import array
from contextlib import ExitStack

from .common import load_specification, usermsgs, MaudeFileFinder
from .statistical import QueryData, check_interval, get_statistical_method, make_parameter_dicts, initialize_methods

# Regular expression for Maude inclusions
LOAD_REGEX = re.compile(rb'^\s*s?load\s+("((?:[^"\\]|\\.)+)"|\S+)')
EOF_REGEX = re.compile(rb'^\s*eof($|\s)')
TOP_COMMENT_REGEX = re.compile(rb'(?:\*{3}|-{3})\s*(.)')
NESTED_COMMENT_REGEX = re.compile(rb'[\(\)]')

# Other constants
ANSWER_SIZE = 2 * ctypes.sizeof(ctypes.c_double) + ctypes.sizeof(ctypes.c_int)


def strip_comments(fobj):
	"""Strip Maude comments from a file"""

	comment_depth = 0  # depth of nested multiline comments
	chunks = []

	for line in fobj:
		start = 0

		while True:
			# Inside a multiline comment
			if comment_depth > 0:
				# Find a the start or end of a multiline comment
				if m := NESTED_COMMENT_REGEX.search(line, start):
					if m.group(0) == b')':  # end
						comment_depth -= 1

					else:  # start
						comment_depth += 1

					start = m.end()

				# The whole line is inside a multiline comment
				else:
					break

			# Not inside a comment
			else:
				if m := TOP_COMMENT_REGEX.search(line, start):
					chunks.append(line[start:m.start()] + b'\n')

					# Start of multiline comment
					if m.group(1) == b'(':
						start = m.end()
						comment_depth += 1

					# Single line comment
					else:
						break

				# No comment at all
				else:
					chunks.append(line[start:])
					break

		if chunks:
			yield b' '.join(chunks)
			chunks.clear()


def flatten_maude_file(filename, fobj):
	"""Scan sources for its dependencies"""

	# Maude file finder
	mfinder = MaudeFileFinder()

	# Files are explored depth-first
	stack = [(lambda v: (filename, v, strip_comments(v)))(open(filename, 'rb'))]
	# Files already included to avoid double inclusion (loads are interpreted as sloads)
	seen = {os.path.realpath(filename)}

	while stack:
		fname, file, lines = stack[-1]

		while line := next(lines, None):
			# Load commands are only allowed at the beginning of a line
			if m := LOAD_REGEX.match(line):
				# File name (with quotes, maybe)
				next_fname = m.group(1).decode()

				if next_fname[0] == '"':
					next_fname = next_fname[1:-1].replace(r'\"', '"')

				# Inclusions are tricky
				next_path, is_std = mfinder.find(next_fname, os.path.dirname(fname))

				# For standard files, we preserve the inclusion
				if is_std:
					fobj.write(line)

				# Otherwise, we copy the file unless already done
				elif next_path not in seen:
					next_file = open(next_path, 'rb')
					stack.append((next_path, next_file, strip_comments(next_file)))
					seen.add(next_path)
					break

			elif EOF_REGEX.match(line):
				line = None
				break
			else:
				fobj.write(line)

		# Whether the file is exhausted
		if line is None:
			fobj.write(b'\n')  # just in case there is no line break at end of file
			stack.pop()
			file.close()


def process_dspec(dspec, fname):
	"""Normalize a distributed SMC specification"""

	if not isinstance(dspec, dict):
		usermsgs.print_error_file('the distribution specification must be a dictionary.', fname)
		return False

	# Normalize workers to a dictionary
	workers = dspec.get('workers')

	if not isinstance(workers, list):
		usermsgs.print_error_file('the distribution specification does not contain a list-valued \'workers\' key.', fname)
		return False

	for k, worker in enumerate(workers):
		# Strings address:port are allowed
		if isinstance(worker, str):
			try:
				address, port = worker.split(':')
				worker = {'address': address, 'port': int(port)}

			except ValueError:
				usermsgs.print_error_file(f'bad address specification {worker} for worker {k + 1} '
				                          '(it should be <address>:<port>).', fname)
				return False

			workers[k] = worker

		# Otherwise, it must be a dictionary
		elif not isinstance(worker, dict):
			usermsgs.print_error_file(f'the specification for worker {k + 1} is not a dictionary.', fname)
			return False

		# With address and port keys
		else:
			for key, ktype in (('address', str), ('port', int)):
				if key not in worker:
					usermsgs.print_error_file(f'missing key \'{key}\' for worker {k + 1}.', fname)
					return False

				if not isinstance(worker[key], ktype):
					usermsgs.print_error_file(f'wrong type for key \'{key}\' in worker {k + 1}, {ktype.__name__} expected.', fname)
					return False

		# Name just for reference in errors and messages
		if 'name' not in worker:
			worker['name'] = f'{worker["address"]}:{worker["port"]}'

	return True


def setup_workers(args, initial_data, dspec, constants, seen_files, stack):
	"""Setup workers and send problem data"""

	workers = dspec['workers']

	# Generate a random seed for each worker
	random.seed(args.seed)

	seeds = [random.getrandbits(20) for _ in range(len(workers))]

	# Data to be passed to the external machine
	COPY = ('initial', 'strategy', 'module', 'metamodule', 'opaque', 'full_matchrew',
	        'purge_fails', 'merge_states', 'assign', 'block', 'query', 'assign', 'advise', 'verbose')

	data = {key: args.__dict__[key] for key in COPY} | {'file': 'source.maude', 'constants': constants, 'dump': args.dump is not None}

	# Make a flattened version of the Maude file
	flat_source = io.BytesIO()
	flatten_maude_file(initial_data.filename, flat_source)

	flat_info = tarfile.TarInfo('source.maude')
	flat_info.size = flat_source.getbuffer().nbytes

	# Save the sockets for each worker
	sockets = []

	# Root of the QuaTEx sources
	quatex_root = os.path.commonpath([os.path.dirname(fn) for fn in seen_files])
	data['query'] = os.path.relpath(data['query'], start=quatex_root)

	for worker, seed in zip(workers, seeds):
		address, port = worker['address'], worker['port']

		try:
			sock = socket.create_connection((address, int(port)))

		except ConnectionRefusedError:
			usermsgs.print_error(f'Connection refused by worker \'{worker["name"]}\'.')
			return None

		stack.enter_context(sock)
		sockets.append(sock)

		# Send the input data
		input_data = data | {'seed': seed}

		if block_size := worker.get('block'):
			input_data['block'] = block_size  # if specified

		input_data = json.dumps(input_data).encode()
		sock.sendall(len(input_data).to_bytes(4, 'big') + input_data)

		# Send the relevant files
		with sock.makefile('wb', buffering=0) as fobj:
			with tarfile.open(mode='w|gz', fileobj=fobj) as tarf:
				flat_source.seek(0)
				tarf.addfile(flat_info, flat_source)

				for file in seen_files:
					relpath = os.path.relpath(file, start=quatex_root)
					tarf.add(file, arcname=relpath)

			fobj.flush()

		# Receive confirmation from the remote
		answer = sock.recv(1)

		if answer != b'o':
			usermsgs.print_error(f'Configuration error in {worker["name"]} worker.')
			return None

	return sockets


def download_dumps(dump_path, workers, nqueries):
	"""Download the dumps from the workers"""

	if parent := os.path.dirname(dump_path):
		os.makedirs(parent, exist_ok=True)

	dump = f'{dump_path}{os.getpid()}-'

	for info in workers:
		sock = info.fileobj
		# Skip unread commands
		while (answer := sock.recv(1)) and answer != b'd':
			if answer == b'b':
				# Discards the block, taking it into account would be better
				sock.recv(ANSWER_SIZE * nqueries)
			else:
				usermsgs.print_error(f'Cannot recover the dump from {sock}')
				break

		# d announces the dump
		if answer == b'd':
			with open(f'{dump}{info.data.name.replace("/", "_")}', 'wb') as dfile:
				while chunk := sock.recv(65495):
					dfile.write(chunk)


class WorkerData:
	"""Relevant data for the worker"""

	__attrs__ = ('index', 'block', 'name', 'requested')

	def __init__(self, index, block, data):
		self.index = index
		self.block = data.get('block', block)
		self.name = data.get('name')
		self.requested = self.block


class FixedGoalScheduler:
	"""Scheduler for fixed goal"""

	def __init__(self, scheduler, count):
		self.scheduler = scheduler
		self.initial = divmod(next(scheduler, 0), count)  # initial block
		self.next_goal = next(scheduler, 0)  # next objective
		self.count = count  # number of workers
		self.pending_requests = self.next_goal  # pending requests for the current goal
		self.pending_workers = count  # pending workers in this round
		self.rounds = [1] * count  # round of each worker
		self.current_round = 1
		self.inactive_workers = []  # inactive workers which have completed their jobs

	def initial_block(self, worker):
		"""Get the initial block for the worker"""

		return self.initial[0] + (worker < self.initial[1])

	def _step_round(self):
		"""Go to the next round"""

		self.next_goal = next(self.scheduler, 0)
		self.pending_requests += self.next_goal
		self.pending_workers = self.count
		self.current_round += 1

	def next_block(self, key):
		"""Get next block size of worker"""

		index = key.data.index

		# Last worker in the round
		if self.pending_workers == 1:
			next_block = self.pending_requests

			if self.next_goal:
				self._step_round()

		else:
			# Assign a share of the current goal submissions
			next_block, rest = divmod(self.pending_requests, self.pending_workers)

			if index < rest:
				next_block += 1

			self.pending_requests -= next_block
			self.pending_workers -= 1

		# We need to store the round of each worker to known
		# whether a failed one is counted in pending_workers or not
		self.rounds[index] = self.current_round

		# Mark the worker as inactive if not given work
		if next_block == 0:
			self.inactive_workers.append(key)

		return next_block

	def worker_lost(self, wdata):
		"""Notify a worker has been lost"""

		# One less worker is available
		self.count -= 1

		# If there are inactive workers, let one of them take the job
		if self.inactive_workers:
			substitute = self.inactive_workers.pop()
			substitute.requested = wdata.requested
			return substitute

		# Otherwise, accumulate the uncompleted work of this worker
		self.pending_requests += wdata.requested

		# If the worker was still pending, it is not now
		if self.current_round > self.rounds[wdata.index]:
			self.pending_workers -= 1


def distributed_check(args, initial_data, min_sim, max_sim, program, constants, seen_files):
	"""Distributed statistical model checking"""

	# Load the distribution specification
	if (dspec := load_specification(args.distribute, 'distribution specification')) is None \
	   or not process_dspec(dspec, args.distribute):
		return None, None

	# Gather all sockets in a context to close them when we finish
	with ExitStack() as stack:

		# Socket to connect with the workers
		if not (sockets := setup_workers(args, initial_data, dspec, constants, seen_files, stack)):
			return None, None

		print('All workers are ready. Starting...', file=sys.stderr)

		# Query data
		qdata = [QueryData(k, delta, idict, method)
		         for k, qinfo in enumerate(program.queries)
		         for idict, delta in make_parameter_dicts(qinfo, args.delta)
		         if (method := get_statistical_method(qinfo.method if qinfo.method else args.smethod, qinfo if qinfo.method else None))]

		# All queries have been discarded
		if not qdata:
			usermsgs.print_error('No valid queries in the input file.')
			return None, None

		# In the distributed setting, the block argument is understood as the block size of
		# each worker, and the configuration may override this value individually
		scheduler = initialize_methods(qdata, args.delta, args.alpha, args.block, min_sim, max_sim)

		# Simulation blocks either follow the block size (if given or required by the method, i.e. if
		# scheduler.block is not null) or equally share the required number of simulation for the next goal
		fixed_goal = FixedGoalScheduler(scheduler, len(sockets)) if scheduler.block is None else None

		# Use a selector to wait for updates from any worker
		selector = selectors.DefaultSelector()

		# Use a selector to be notified of the worker answers
		for k, (sock, data) in enumerate(zip(sockets, dspec['workers'])):
			wdata = WorkerData(k, scheduler.block, data)
			selector.register(sock, selectors.EVENT_READ, data=wdata)

			# Adjust the initial amount for the fixed number case
			if fixed_goal:
				wdata.requested = fixed_goal.initial_block(k)

			# Send the initial amount of simulations
			sock.send(b'c' + wdata.requested.to_bytes(4, signed=False))  # continue command

		buffer = array('d')
		ibuffer = array('i')

		nqueries = len(qdata)
		num_sims = 0

		# In order to tell the workers which queries have converged, we keep
		# a list of indices of converged queries in cronological order and
		# a list with the next index to be transmitted to each worker
		converged_queries = array('i')
		next_to_tell = [0] * len(sockets)

		def command_new_block(key):
			"""Command a new job to the given worker (inline definition)"""

			index = key.data.index

			# Transmit which queries have converged
			if next_to_tell[index] != len(converged_queries):
				# We send p followed by the number of indices and then the indices
				key.fileobj.send(b'p'
					+ (len(converged_queries) - next_to_tell[index]).to_bytes(4, 'big')
					+ converged_queries[next_to_tell[index]:].tobytes())
				next_to_tell[index] = len(converged_queries)

			# Continue without change
			else:
				key.fileobj.send(b'c')

			key.fileobj.send(key.data.requested.to_bytes(4, signed=False))

		while sockets:
			events = selector.select()
			finished = []

			for key, _ in events:
				sock = key.fileobj

				answer = sock.recv(1)

				# Block finished
				if answer == b'b':
					# The message is the concatenation of three arrays
					# (sum, sum_sq, and counts) of nqueries elements each.
					# sum and sum_sq contain 64-bit floating-point numbers
					# and counts contains 32-bit integer numbers
					data = sock.recv(ANSWER_SIZE * nqueries)
					buffer.frombytes(data[:16 * nqueries])
					ibuffer.frombytes(data[16 * nqueries:])

					# Some queries may have not been evaluated because
					# they converged, but it is easier to sum their zeros
					for k in range(nqueries):
						qdata[k].sum += buffer[k]
						qdata[k].sum_sq += buffer[nqueries + k]
						qdata[k].n += ibuffer[k]

					num_sims += key.data.requested

					del buffer[:]
					del ibuffer[:]
					finished.append(key)

				else:
					# Its work should be done by the others
					if fixed_goal:
						substitute = fixed_goal.worker_lost(key.data)
						command_new_block(substitute)

					# Inform and remove it from the lists of active workers
					usermsgs.print_error(f'Server {key.data.name} disconnected or misbehaving')
					selector.unregister(key.fileobj)
					sockets.remove(key.fileobj)


			# Check whether the simulation has converged
			converged, which = check_interval(qdata, num_sims, min_sim, args.alpha, args.verbose)

			# More converged queries
			if which:
				converged_queries.extend(which)

			if converged or not sockets:
				break

			for key in finished:
				if fixed_goal:
					key.data.requested = fixed_goal.next_block(key)

					# Skip the request if no simulations are pending
					if key.data.requested == 0:
						continue

				command_new_block(key)

			finished.clear()

		# Send stop signal to all workers
		for sock in sockets:
			sock.send(b's')

		# If dump were requested, receive those from the workers
		if args.dump:
			download_dumps(args.dump, selector.get_map().values(), nqueries)

	return num_sims, qdata
