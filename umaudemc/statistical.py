#
# Statistical model-checking engine
#

import contextlib
import math
import os
import random

from .common import usermsgs, maude


def get_quantile_func():
	"""Get the quantile function for calculating confidence intervals"""

	try:
		# SciPy is required for obtaining the quantiles of the Student's
		# t-distribution, although it is a huge dependency including NumPy
		from scipy.special import stdtrit
		return stdtrit

	except ImportError:
		usermsgs.print_warning('Warning: using the normal distribution for calculating '
		                       'confidence intervals instead of Student\'s t-distribution, '
		                       'since SciPy is not available.')

		from statistics import NormalDist
		normal = NormalDist()
		# The first argument is the degrees of freedom of the
		# Student's t-distribution, so it is ignored here
		return lambda _, p: normal.inv_cdf(p)


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

	def __init__(self, query, delta, params):
		# Query expression index
		self.query = query
		# Initial dictionary of variable values
		self.params = params
		# Radius of the confidence interval
		self.delta = delta

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


def make_parameter_dicts(qinfo):
	"""Make the initial variable mapping for the parameters of a query"""

	if qinfo is None:
		yield {}

	else:
		var, x, step, end = qinfo

		while x <= end:
			yield {var: x}
			x += step


def check_interval(qdata, num_sims, min_sim, alpha, quantile, verbose):
	"""Check the confidence interval"""

	# Whether the size of the confidence interval for all queries have converged
	converged = True

	for query in qdata:
		# This query has already converged
		if query.converged:
			continue
		# All executions of this query have been discarded
		elif query.n == 0:
			converged = False
			continue

		# The radius encloses the confidence level in the reference
		# distribution for calculating confidence intervals
		tinv = quantile(query.n - 1, 1 - alpha / 2) / math.sqrt(query.n)

		query.mu = query.sum / query.n
		query.s = math.sqrt(max(query.sum_sq - query.sum * query.mu, 0.0) / (query.n - 1))
		query.h = query.s * tinv

		if query.h <= query.delta and query.n >= min_sim:
			query.converged = True
			query.discarded = num_sims - query.n
		else:
			converged = False

	# Print intermediate results if in verbose mode
	if verbose:
		usermsgs.print_info(f'  step={num_sims} n={" ".join(str(q.n) for q in qdata)}'
		                    f' μ={" ".join(str(q.mu) for q in qdata)}'
		                    f' σ={" ".join(str(q.s) for q in qdata)}'
		                    f' r={" ".join(str(q.h) for q in qdata)}')

	return converged


def run_single(program, qdata, num_sims, min_sim, max_sim, simulator, alpha, block_size,
               verbose=False, dump=None):
	"""Run simulation in a single thread"""

	# Size of the first block of execution (it coincides with num_sims
	# but these number will differ if more blocks are executed)
	block = num_sims

	quantile = get_quantile_func()

	# This loop continues until the maximum number of simulation is reached
	# or the confidence interval converges to the given delta
	while True:
		for _ in range(block):
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

		converged = check_interval(qdata, num_sims, min_sim, alpha, quantile, verbose)

		if converged or max_sim and num_sims >= max_sim:
			break

		num_sims += block_size
		block = block_size

	return num_sims, qdata


def thread_main(program, qdata, simulator, num_sims, block_size, seed, queue, barrier, more, dump=None):
	"""Entry point of a calculating thread"""

	maude.setRandomSeed(seed)
	random.seed(seed)

	block = num_sims

	# Open dump file for the raw data
	dump_file = open(dump, 'w') if dump else None

	sums = [0.0] * len(qdata)
	sum_sq = [0.0] * len(qdata)
	counts = [0] * len(qdata)

	# Repeat until the main process says we are done
	while True:

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

		if not more.value:
			break

		# Continue for a next block
		block = block_size


def run_parallel(program, qdata, num_sims, min_sim, max_sim, simulator, alpha, block_size, jobs, verbose=False, dump=None):
	"""Run the simulation in multiple threads"""
	import multiprocessing as mp
	mp.set_start_method('fork', force=True)

	# When the number of jobs is zero or negative, we take the CPU count
	if jobs <= 0:
		jobs = os.cpu_count()

	# Like in run_single
	quantile = get_quantile_func()

	# Process communication stuff

	# Random number seeds
	seeds = [random.getrandbits(20) for _ in range(jobs)]
	# Dump file names
	dumps = [f'{dump}.{os.getpid()}-{k}' for k in range(jobs)] if dump else ([None] * jobs)
	# Queue for transferring the query evaluations
	queue = mp.Queue()
	barrier = mp.Barrier(jobs + 1)
	more = mp.Value('b', False, lock=False)

	rest, rest_block = num_sims % jobs, block_size % jobs
	processes = [mp.Process(target=thread_main,
	                        args=(program, qdata, simulator, num_sims // jobs + (k < rest),
	                              block_size // jobs + (k < rest_block),
	                              seeds[k], queue, barrier, more, dumps[k])) for k in range(jobs)]

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

		converged = check_interval(qdata, num_sims, min_sim, alpha, quantile, verbose)

		if converged or max_sim and num_sims >= max_sim:
			break

		num_sims += block_size

		more.value = True
		barrier.wait()

	more.value = False
	barrier.wait()

	# Wait for all processes
	for p in processes:
		p.join()

	return num_sims, qdata


def qdata_to_dict(num_sims, qdata, program):
	"""Convert the raw output of the statistical model checker to a dictionary"""

	queries = []
	qdata_it = iter(qdata)
	q = next(qdata_it, None)

	for k, (fname, line, column, params) in enumerate(program.query_locations):
		# For parametric queries, we return an array of values
		if params:
			mean, std, radius, count, discarded = [], [], [], [], []

			while q and q.query == k:
				mean.append(q.mu)
				std.append(q.s)
				radius.append(q.h)
				count.append(q.n)
				discarded.append(q.discarded)
				q = next(qdata_it, None)

			# We also write information about the parameter
			param_info = {'params': [dict(name=params[0], start=params[1], step=params[2], stop=params[3])]}

		else:
			mean, std, radius, count, discarded = q.mu, q.s, q.h, q.n, q.discarded
			param_info = {}

		queries.append(dict(mean=mean, std=std, radius=radius, file=fname, line=line, column=column,
		                    nsims=count, discarded=discarded, **param_info))

	return dict(nsims=num_sims, queries=queries)


def check(program, simulator, seed, alpha, delta, block, min_sim, max_sim, jobs, verbose=False, dump=None):
	"""Run the statistical model checker"""

	# The number of simulations for the first block
	num_sims = max(block, min_sim or 1)

	if max_sim:
		num_sims = min(num_sims, max_sim)

	# Set the random seed (both Python's and Maude's)
	if seed is not None and seed >= 0:
		maude.setRandomSeed(seed)
		random.seed(seed)

	# Each query maintains some data like the sum of the outcomes
	# and the sum of their squares
	qdata = [QueryData(k, delta, idict)
	         for k, qinfo in enumerate(program.query_locations)
	         for idict in make_parameter_dicts(qinfo[3])]

	# Run the simulations
	if jobs == 1 and num_sims != 1:
		with (open(dump, 'w') if dump else contextlib.nullcontext()) as dump_file:
			return run_single(program, qdata, num_sims, min_sim, max_sim, simulator, alpha,
					  block, verbose=verbose, dump=dump_file)
	else:
		return run_parallel(program, qdata, num_sims, min_sim, max_sim, simulator, alpha,
				    block, jobs, verbose=verbose, dump=dump)
