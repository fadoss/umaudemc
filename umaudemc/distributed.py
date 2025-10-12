#
# Distributed model checking
#

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
from .statistical import QueryData, check_interval, get_quantile_func, make_parameter_dicts

# Regular expression for Maude inclusions
LOAD_REGEX = re.compile(rb'^\s*s?load\s+("((?:[^"\\]|\\.)+)"|\S+)')
EOF_REGEX = re.compile(rb'^\s*eof($|\s)')
TOP_COMMENT_REGEX = re.compile(rb'(?:\*{3}|-{3})\s*(.)')
NESTED_COMMENT_REGEX = re.compile(rb'[\(\)]')


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

	data = {key: args.__dict__[key] for key in COPY} | {'file': 'source.maude', 'constants': constants}

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
		sock.sendall(len(input_data).to_bytes(4) + input_data)

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

		# Use a selector to wait for updates from any worker
		selector = selectors.DefaultSelector()

		for sock, data in zip(sockets, dspec['workers']):
			selector.register(sock, selectors.EVENT_READ, data={'block': args.block} | data)
			sock.send(b'c')

		buffer = array('d')
		ibuffer = array('i')

		# Query data
		qdata = [QueryData(k, args.delta, idict)
		         for k, qinfo in enumerate(program.query_locations)
		         for idict in make_parameter_dicts(qinfo[3])]
		nqueries = len(qdata)
		num_sims = 0

		quantile = get_quantile_func()

		while sockets:
			events = selector.select()
			finished = []

			for key, _ in events:
				sock = key.fileobj

				answer = sock.recv(1)

				if answer == b'b':
					data = sock.recv(24 * nqueries)
					buffer.frombytes(data[:16 * nqueries])
					ibuffer.frombytes(data[16 * nqueries:])

					for k in range(nqueries):
						qdata[k].sum += buffer[k]
						qdata[k].sum_sq += buffer[nqueries + k]
						qdata[k].n += ibuffer[k]

					num_sims += key.data['block']

					del buffer[:]
					del ibuffer[:]
					finished.append(key.fileobj)

				else:
					usermsgs.print_error(f'Server {key.data["name"]} disconnected or misbehaving')
					selector.unregister(key.fileobj)
					sockets.remove(key.fileobj)

			# Check whether the simulation has converged
			converged = check_interval(qdata, num_sims, min_sim, args.alpha, quantile, args.verbose)

			if converged or max_sim and num_sims >= max_sim:
				break

			for sock in finished:
				sock.send(b'c')

			finished.clear()

		# Send stop signal to all workers
		for sock in sockets:
			sock.send(b's')

	return num_sims, qdata
