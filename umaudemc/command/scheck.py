#
# Command-line statistical model-checking subcommand
#

import math
import os
import random

from ..common import parse_initial_data, usermsgs, maude
from ..quatex import parse_quatex
from ..simulators import get_simulator


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


def run(program, simulator):
	"""Run a program on a simulator"""

	simulator.restart()

	# Program counter (slot index) for each query
	pc = list(range(program.ndefs, len(program.slots)))
	# List where to store the results of each query
	results = [None] * program.nqueries
	# Remaining queries for being processed
	remaining = program.nqueries

	# Variables when evaluating expressions
	cvars = [{'rval': simulator.rval}] * program.nqueries

	# Keep running until all queries have been calculated
	while True:
		# Every query is evaluated in the current simulation state
		for k in range(program.nqueries):
			# Expressions may call functions with or without the next
			# operator. The following keep running slots until a call
			# with next is encountered
			has_next = False

			while not has_next:
				# Execute the compiled slot
				value = eval(program.slots[pc[k]], cvars[k])

				# The evaluation of the k-th query has finished
				if isinstance(value, float):
					remaining -= 1
					results[k] = value

					# All queries have been calculated
					if not remaining:
						return results

					break

				# The expression finishes with a call
				else:
					has_next, jump, *args = value

					pc[k] = jump
					cvars[k] = dict(zip(program.varnames[pc[k]], args),
					                rval=simulator.rval)

		# Execute a step of the simulation
		simulator.next_step()


class QueryData:
	"""Data associated to a query under evaluation"""

	def __init__(self):
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


def check_interval(qdata, num_sims, alpha, delta, quantile, verbose):
	"""Check the confidence interval"""

	# The radius of encloses the confidence level in the reference
	# distribution for calculating confidence intervals
	tinv = quantile(num_sims - 1, 1 - alpha / 2) / math.sqrt(num_sims)

	# Whether the size of the confidence interval for all queries have converged
	converged = True

	for k, query in enumerate(qdata):
		query.mu = query.sum / num_sims
		query.s = math.sqrt((query.sum_sq - query.sum * query.mu) / (num_sims - 1))
		query.h = query.s * tinv

		if query.h > delta:
			converged = False

	# Print intermediate results if in verbose mode
	if verbose:
		usermsgs.print_info(f'  step={num_sims} μ={" ".join(str(q.mu) for q in qdata)}'
		                    f' σ={" ".join(str(q.s) for q in qdata)}'
		                    f' r={" ".join(str(q.h) for q in qdata)}')

	return converged


def run_single(program, num_sims, max_sim, simulator, alpha, delta, block_size, verbose=False):
	"""Run simulation in a single thread"""

	# Size of the first block of execution (it coincides with num_sims
	# but these number will differ if more blocks are executed)
	block = num_sims

	quantile = get_quantile_func()

	# Each query maintains some data like the sum of the outcomes
	# and the sum of their squares
	qdata = [QueryData() for _ in range(program.nqueries)]

	# This loop continues until the maximum number of simulation is reached
	# or the confidence interval converges to the given delta
	while True:
		for _ in range(block):
			# Run the simulation and compute all queries at once
			values = run(program, simulator)

			for k, query in enumerate(qdata):
				query.sum += values[k]
				query.sum_sq += values[k] * values[k]

		converged = check_interval(qdata, num_sims, alpha, delta, quantile, verbose)

		if converged or max_sim and num_sims >= max_sim:
			break

		num_sims += block_size
		block = block_size

	return num_sims, qdata


def thread_main(program, simulator, num_sims, block_size, seed, queue, barrier, more):
	"""Entry point of a calculating thread"""

	maude.setRandomSeed(seed)
	random.seed(seed)

	block = num_sims

	sums = [0.0] * program.nqueries
	sum_sq = [0.0] * program.nqueries

	# Repeat until the main process says we are done
	while True:

		for _ in range(block):
			# Run the simulation and compute all queries at once
			values = run(program, simulator)

			for k in range(program.nqueries):
				sums[k] += values[k]
				sum_sq[k] += values[k] * values[k]

		# Send the results to the main process and wait for it
		queue.put((sums, sum_sq))
		barrier.wait()

		for k in range(program.nqueries):
			sums[k] = 0.0
			sum_sq[k] = 0.0

		if not more.value:
			break

		# Continue for a next block
		block = block_size


def run_parallel(program, num_sims, max_sim, simulator, alpha, delta, block_size, jobs, verbose=False):
	"""Run the simulation in multiple threads"""
	import multiprocessing as mp

	# When the number of jobs is zero or negative, we take the CPU count
	if jobs <= 0:
		jobs = os.cpu_count()

	# Like in run_single
	qdata = [QueryData() for _ in range(program.nqueries)]
	quantile = get_quantile_func()

	# Process communication stuff

	# Random number seeds
	seeds = [random.randint(0, 1e6) for _ in range(jobs)]
	# Queue for transferring the query evaluations
	queue = mp.Queue()
	barrier = mp.Barrier(jobs + 1)
	more = mp.Value('b', False, lock=False)

	rest, rest_block = num_sims % jobs, block_size % jobs
	processes = [mp.Process(target=thread_main,
	                        args=(program, simulator, num_sims // jobs + (k < rest),
	                              block_size // jobs + (k < rest_block),
	                              seeds[k], queue, barrier, more)) for k in range(jobs)]

	# Start all processes
	for p in processes:
		p.start()

	# Exactly as in run_single but with several threads
	while True:
		for j in range(jobs):
			sums, sum_sq = queue.get()

			for k, query in enumerate(qdata):
				query.sum += sums[k]
				query.sum_sq += sum_sq[k]

		converged = check_interval(qdata, num_sims, alpha, delta, quantile, verbose)

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


def show_results(program, nsims, qdata):
	"""Show the results of the calculation"""

	print(f'Number of simulations = {nsims}')

	for k, query in enumerate(qdata):
		if program.nqueries > 1:
			line, column = program.query_locations[k]
			print(f'Query {k + 1} (line {line}:{column})')
		print(f'  μ = {query.mu:<25} σ = {query.s:<25} r = {query.h}')


def show_json(program, nsims, qdata):
	"""Show the results of the calculation in JSON"""

	print(f'{{"nsims": {nsims}, "queries": [')

	for k, query in enumerate(qdata):
		if k > 0:
			print(',')
		line, column = program.query_locations[k]
		print(f'  {{"mean": {query.mu}, "std": {query.s}, "radius": {query.h}, '
	              f'"line": {line}, "column": {column}}}', end='')

	print(']}')


def parse_range(rtext):
	"""Parse a min-max range with potential omissions"""

	try:
		lims = tuple((int(arg) if arg else None) for arg in rtext.split('-'))

	except ValueError as ve:
		usermsgs.print_error(f'invalid range: {ve}.')
		return None, None

	if len(lims) > 2:
		usermsgs.print_error('too many steps in the simulation range.')
		return None, None

	if len(lims) == 1:
		return lims[0], lims[0]

	return lims


def scheck(args):
	"""Statistical check subcommand"""

	# Parse the model-checking problem data
	data = parse_initial_data(args)

	if data is None:
		return 1

	# Parse the QuaTEx query
	if not os.path.exists(args.query):
		usermsgs.print_error(f'The query file "{args.query}" does not exist.')
		return 1

	program = parse_quatex(args.query)

	if not program:
		return 1

	# Get the simulator for the given assignment method
	simulator = get_simulator(args.assign, data)

	if not simulator:
		return 1

	# Check the simulation parameters

	if not (0 <= args.alpha <= 1):
		usermsgs.print_error(f'Wrong value {args.alpha} for the alpha parameter (must be between 0 and 1).')
		return 1

	if args.delta < 0:
		usermsgs.print_error(f'Wrong value {args.delta} for the delta parameter (must be positive).')
		return 1

	if args.block <= 0:
		usermsgs.print_error(f'Wrong block size {args.block} (must be positive).')
		return 1

	min_sim, max_sim = parse_range(args.nsims)

	if min_sim is None:
		return 1

	# The number of simulations for the first block
	num_sims = max(args.block, min_sim or 1)

	if max_sim:
		num_sims = min(num_sims, max_sim)

	# Set the random seed (both Python's and Maude's)
	if args.seed is not None and args.seed >= 0:
		maude.setRandomSeed(args.seed)
		random.seed(args.seed)

	# Run the simulations
	if args.jobs == 1:
		num_sims, qdata = run_single(program, num_sims, max_sim, simulator, args.alpha,
		                             args.delta, args.block, verbose=args.verbose)
	else:
		num_sims, qdata = run_parallel(program, num_sims, max_sim, simulator, args.alpha,
		                               args.delta, args.block, args.jobs, verbose=args.verbose)

	# Print the results on the terminal
	(show_json if args.format == 'json' else show_results)(program, num_sims, qdata)

	return 0
