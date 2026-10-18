#
# Statistical model-checking engine
#

import contextlib
import math
import os
import random

from .common import usermsgs, maude
from .smethods import get_statistical_method


def run(program, qdata, simulator):
	"""Run a program on a simulator"""

	simulator.restart()

	# Program counter (slot index) for each query
	pc = [program.ndefs + q.query for q in qdata]
	# List where to store the results of each query
	results = [None] * len(qdata)
	# Remaining queries for being processed
	remaining = [k for k, q in enumerate(qdata) if not q.converged]

	# Variables when evaluating expressions
	cvars = [{'rval': simulator.rval, **q.params} for q in qdata]

	# Keep running until all queries have been calculated
	while True:
		# Every query is evaluated in the current simulation state
		index = 0

		while index < len(remaining):
			k = remaining[index]

			# Expressions may call functions with or without the next
			# operator. The following keep running slots until a call
			# with next is encountered
			has_next = False

			while not has_next:
				# Execute the compiled slot
				value = eval(program.slots[pc[k]], cvars[k])

				# The expression finishes with a call
				if isinstance(value, tuple):
					has_next, jump, *args = value

					pc[k] = jump
					cvars[k] = dict(zip(program.varnames[pc[k]], args),
					                rval=simulator.rval)

				# The evaluation of the k-th query has finished
				else:
					remaining.pop(index)
					results[k] = value

					# All queries have been calculated
					if not remaining:
						return results

					break


			if has_next:
				index += 1

		# Execute a step of the simulation
		simulator.next_step()


class QueryData:
	"""Data associated to a query under evaluation"""

	def __init__(self, query, delta, params, smethod):
		# Query expression index
		self.query = query
		# Initial dictionary of variable values
		self.params = params
		# Radius of the confidence interval
		self.delta = math.inf if delta is None else delta
		# Statistical method used for calculating the confidence interval
		self.smethod = smethod

		# Sum of the query outcomes
		self.sum = 0.0
		# Sum of the squares of the query outcomes
		self.sum_sq = 0.0
		# Mean of the sample
		self.mu = 0.0
		# Standard deviation of the sample
		self.s = 0.0
		# Radius of the confidence interval
		self.h = 0.0
		# Number of runs
		self.n = 0
		# Whether the query has converged
		self.converged = False

		# Number of discarded runs
		self.discarded = 0


def make_parameter_dicts(qinfo, delta):
	"""Make the initial variable mapping for the parameters of a query"""

	if qinfo.parameters is None:
		yield {}, (qinfo.delta() if qinfo.delta else delta)

	else:
		var, x, step, end = qinfo.parameters

		while x <= end:
			yield {var: x}, (qinfo.delta(x) if qinfo.delta else delta)
			x += step


def check_interval(qdata, num_sims, min_sim, alpha, verbose):
	"""Check the confidence interval"""

	# Whether the size of the confidence interval for all queries have converged
	converged = True
	# Which queries have converged
	which = []

	for k, query in enumerate(qdata):
		# This query has already converged
		if query.converged:
			continue
		# All executions of this query have been discarded
		elif query.n == 0:
			converged = False
			continue
		# A single execution
		elif query.n == 1:
			query.mu, query.s, query.h = query.sum, 0.0, 0.0
		# General case
		else:
			query.mu = query.sum / query.n
			query.s = math.sqrt(max(query.sum_sq - query.sum * query.mu, 0.0) / (query.n - 1))
			query.smethod.check(query, alpha, min_sim)

		if query.converged:
			query.discarded = num_sims - query.n
			which.append(k)
		else:
			converged = False

	# Print intermediate results if in verbose mode
	if verbose:
		usermsgs.print_info(f'  step={num_sims} n={" ".join(str(q.n) for q in qdata)}'
		                    f' μ={" ".join(str(q.mu) for q in qdata)}'
		                    f' σ={" ".join(str(q.s) for q in qdata)}'
		                    f' r={" ".join(str(q.h) for q in qdata)}')

		for k in which:
			print(f'  Query {qdata[k].query + 1} has converged')

	return converged, which


class Scheduler:
	"""Decide the number of executions required in each step"""

	def __init__(self, block, min_sim, max_sim, stops):
		self.block = block  # size of the periodic block
		self.current = 0  # current number of simulations
		self.max_sim = max_sim # maximum number of executions

		# Stops to check convergence
		self.stops = sorted({k for k in stops if k > min_sim}, reverse=True)

		# Number of simulations for the first block
		self.first_block = max(block, min_sim) if block else min_sim

		if max_sim and max_sim < self.first_block:
			self.first_block = max_sim

	def __iter__(self):
		"""The scheduler is an iterator"""

		return self

	def __next__(self):
		"""Get the next stop"""

		# First block
		if self.first_block:
			self.current, self.first_block = self.first_block, None
			return self.current

		# Already done
		if self.max_sim and self.current >= self.max_sim:
			raise StopIteration

		# Using fixed stops
		if self.block is None or (self.stops and self.current + self.block >= self.stops[-1]):
			if not self.stops:
				raise StopIteration

			candidate = self.stops.pop()

		# Using a periodic stop
		else:
			candidate = self.current + self.block

			if self.max_sim and candidate > self.max_sim:
				candidate = self.max_sim

		next_step = candidate - self.current
		self.current = candidate
		return next_step


def run_single(program, qdata, min_sim, scheduler, simulator, alpha,
               verbose=False, dump=None):
	"""Run simulation in a single thread"""

	# This loop continues until the maximum number of simulation is reached
	# or the confidence interval converges to the given delta
	for block_size in scheduler:
		for _ in range(block_size):
			# Run the simulation and compute all queries at once
			values = run(program, qdata, simulator)

			# Dump evaluations if required
			if dump:
				print(*values, file=dump)

			for value, query in zip(values, qdata):
				if value is not None:
					query.sum += value
					query.sum_sq += value * value
					query.n += 1

		converged, _ = check_interval(qdata, scheduler.current, min_sim, alpha, verbose)

		if converged:
			break

	return scheduler.current, qdata


def thread_main(program, qdata, simulator, pindex, num_jobs, seed, queue, barrier, more, convergents, dump=None):
	"""Entry point of a calculating thread"""

	maude.setRandomSeed(seed)
	random.seed(seed)

	# Open dump file for the raw data
	dump_file = open(dump, 'w') if dump else None

	sums = [0.0] * len(qdata)
	sum_sq = [0.0] * len(qdata)
	counts = [0] * len(qdata)

	# Number of simulations for this block
	block = more.value

	# Repeat until the main process says we are done
	while True:
		# Adjust block size for this process
		block = (block // num_jobs) + (pindex < block % num_jobs)

		for _ in range(block):
			# Run the simulation and compute all queries at once
			values = run(program, qdata, simulator)

			if dump is not None:
				print(*values, file=dump_file)

			for k in range(len(qdata)):
				if values[k] is not None:
					sums[k] += values[k]
					sum_sq[k] += values[k] * values[k]
					counts[k] += 1

		# Send the results to the main process and wait for it
		queue.put((sums, sum_sq, counts))
		barrier.wait()

		for k in range(len(qdata)):
			sums[k] = 0.0
			sum_sq[k] = 0.0
			counts[k] = 0

		if more.value == 0:
			break

		# Disable queries that have converged
		for k in range(len(qdata)):
			if convergents[k] == -1:
				break
			qdata[convergents[k]].converged = True

		# Continue for a next block
		block = more.value


def run_parallel(program, qdata, min_sim, scheduler, simulator, alpha, jobs,
                 verbose=False, method='chow-robbins', dump=None):
	"""Run the simulation in multiple threads"""
	import multiprocessing as mp
	mp.set_start_method('fork', force=True)

	# When the number of jobs is zero or negative, we take the CPU count
	if jobs <= 0:
		jobs = os.cpu_count()

	# Process communication stuff

	# Random number seeds
	seeds = [random.getrandbits(20) for _ in range(jobs)]
	# Dump file names
	dumps = [f'{dump}.{os.getpid()}-{k}' for k in range(jobs)] if dump else ([None] * jobs)
	# Queue for transferring the query evaluations
	queue = mp.Queue()
	# Barrier to synchronize processes
	# (running them asynchronously could be an alternative)
	barrier = mp.Barrier(jobs + 1)
	more = mp.Value('i', next(scheduler), lock=False)
	convergents = mp.Array('i', [-1] * len(qdata), lock=False)

	processes = [mp.Process(target=thread_main,
	                        args=(program, qdata, simulator, k, jobs, seeds[k],
	                              queue, barrier, more, convergents, dumps[k]))
	             for k in range(jobs)]

	# Start all processes
	for p in processes:
		p.start()

	# Exactly as in run_single but with several threads
	while True:
		for _ in range(jobs):
			sums, sum_sq, counts = queue.get()

			for k, query in enumerate(qdata):
				query.sum += sums[k]
				query.sum_sq += sum_sq[k]
				query.n += counts[k]

		converged, which = check_interval(qdata, scheduler.current, min_sim, alpha, verbose)

		if converged:
			break

		# Obtain the next step of executions from the scheduler
		more.value = next(scheduler, 0)

		if more.value == 0:
			break

		# Inform which queries have converged
		for k, qindex in enumerate(which):
			convergents[k] = qindex
		convergents[len(which)] = -1

		barrier.wait()

	more.value = 0
	barrier.wait()

	# Wait for all processes
	for p in processes:
		p.join()

	return scheduler.current, qdata


def qdata_to_dict(num_sims, qdata, program):
	"""Convert the raw output of the statistical model checker to a dictionary"""

	queries = []
	qdata_it = iter(qdata)
	q = next(qdata_it, None)

	for k, query in enumerate(program.queries):
		# For parametric queries, we return an array of values
		if query.parameters:
			mean, std, radius, count, discarded = [], [], [], [], []

			while q and q.query == k:
				mean.append(q.mu)
				std.append(q.s)
				radius.append(q.h)
				count.append(q.n)
				discarded.append(q.discarded)
				q = next(qdata_it, None)

			# We also write information about the parameter
			param_info = {'params': [dict(zip(('name', 'start', 'step', 'stop'), query.parameters))]}

		else:
			mean, std, radius, count, discarded = q.mu, q.s, q.h, q.n, q.discarded
			q = next(qdata_it, None)
			param_info = {}

		queries.append(dict(mean=mean, std=std, radius=radius,
		                    file=query.filename, line=query.line, column=query.column,
		                    nsims=count, discarded=discarded, **param_info))

	return dict(nsims=num_sims, queries=queries)


def initialize_methods(qdata, delta, alpha, block, min_sim, max_sim, verbose=False, default_block=30):
	"""Initialize the statistical methods of the given queries"""

	max_required, max_query, stops = min_sim, None, []

	for query in qdata:
		# A number of execution is required for this query
		if required := query.smethod.init(query.delta, alpha, min_sim):
			# Insufficient number of executions
			if max_sim and required > max_sim:
				usermsgs.print_warning(f'Query {query.query + 1} requires {required} simulations by {query.smethod.NAME}, '
				                       f'but at most {max_sim} are allowed')

			else:
				stops.append(required)

				if required > max_required:
					max_required = required
					max_query = query

	# Inform about the number of required simulations
	if verbose and max_query:
		usermsgs.print_info(f'Query {max_query.query + 1} requires {max_required} simulations by {max_query.smethod.NAME}')

	# Set a default delta value if sequential methods are used
	if block is None and len(stops) < len(qdata):
		block = default_block

	# Prepare an scheduler for the experiment
	scheduler = Scheduler(block, min_sim, max_sim, stops)

	return scheduler


def check(program, simulator, seed, alpha, delta, block, min_sim, max_sim, jobs, smethod, verbose=False, dump=None):
	"""Run the statistical model checker"""

	# Set the random seed (both Python's and Maude's)
	if seed is not None and seed >= 0:
		maude.setRandomSeed(seed)
		random.seed(seed)

	# Each query maintains some data like the sum of the outcomes
	# and the sum of their squares
	qdata = [QueryData(k, dt, idict, method)
	         for k, qinfo in enumerate(program.queries)
	         for idict, dt in make_parameter_dicts(qinfo, delta)
		 if (method := get_statistical_method(qinfo.method if qinfo.method else smethod,
		                                      qinfo=qinfo if qinfo.method else None))]

	# All queries have been discarded
	if not qdata:
		usermsgs.print_error('No valid queries in the input file.')
		return None, None

	# Initialize statistical methods
	# (get the scheduler to decide when to stop for checking convergence or at all)
	scheduler = initialize_methods(qdata, delta, alpha, block, min_sim, max_sim, verbose=verbose)

	# If a dump path is given, try to create its parent directories if they do not exist
	if dump and (parent := os.path.dirname(dump)):
		try:
			os.makedirs(parent, exist_ok=True)

		except OSError as oe:
			usermsgs.print_error(f'Cannot create parent directories for the dump file: {oe}')
			return None, None

	# Run the simulations
	if jobs == 1 and scheduler.first_block != 1:
		with (open(dump, 'w') if dump else contextlib.nullcontext()) as dump_file:
			return run_single(program, qdata, min_sim, scheduler, simulator, alpha,
					  verbose=verbose, dump=dump_file)
	else:
		return run_parallel(program, qdata, min_sim, scheduler, simulator, alpha, jobs,
				    verbose=verbose, dump=dump)
