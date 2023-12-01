#
# Command-line statistical model-checking subcommand
#

import os

from ..common import parse_initial_data, usermsgs
from ..quatex import parse_quatex
from ..simulators import get_simulator
from ..statistical import check, qdata_to_dict


def show_results(program, nsims, qdata):
	"""Show the results of the calculation"""

	print(f'Number of simulations = {nsims}')

	# Iterator for qdata
	qdata_it = iter(qdata)
	q = next(qdata_it, None)

	for k, (line, column, params) in enumerate(program.query_locations):
		# Print the query name and location only if there are many
		if program.nqueries > 1:
			print(f'Query {k + 1} (line {line}:{column})')

		# For parametric queries, we show the result for every value
		var = params[0] if params else None

		while q and q.query == k:
			if var:
				print(f'  {var} = {q.params[var]:<10} μ = {q.mu:<20} σ = {q.s:<20} r = {q.h}')
			else:
				print(f'  μ = {q.mu:<25} σ = {q.s:<25} r = {q.h}')

			q = next(qdata_it, None)


def show_json(program, nsims, qdata):
	"""Show the results of the calculation in JSON"""

	import json, sys
	json.dump(qdata_to_dict(nsims, qdata, program), sys.stdout)
	print()


def plot_results(program, qdata):
	"""Plot the results of parametric queries"""

	# Gather the results of parametric queries
	xs, ys, rs, index, results = [], [], [], 0, []

	for query in qdata:
		if not query.params:
			continue

		if query.query == index:
			x, = query.params.values()
			xs.append(x)
			ys.append(query.mu)
			rs.append(query.h)

		else:
			# Parametric queries with less than two points are
			# ignored (line plots do not make sense for them)
			if len(xs) > 1:
				results.append((index, xs, ys, rs))
				xs, ys, rs = [], [], []

			index = query.query

	if len(xs) > 1:
		results.append((index, xs, ys, rs))

	if not results:
		usermsgs.print_warning('Skipping plotting since there are no (non-trivial) parametric queries to plot.')
		return

	try:
		import matplotlib.pyplot as plt

	except ImportError:
		usermsgs.print_warning('Skipping plotting since Matplotlib is not available. '
		                       'It can be installed with pip install matplotlib.')
		return

	for k, xs, ys, rs in results:
		line, column, _ = program.query_locations[k]

		# Plot the mean
		p = plt.plot(xs, ys, label=f'{line}:{column}')
		# Plot the confidence interval
		plt.fill_between(xs, [y - r for y, r in zip(ys, rs)],
		                 [y + r for y, r in zip(ys, rs)],
		                 color=p[0].get_color(), alpha=.1)

	# Show a legend when there is more than one query
	if len(results) > 1:
		plt.legend()

	plt.show()


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

	with open(args.query) as quatex_file:
		program = parse_quatex(quatex_file, filename=args.query)

	if not program:
		return 1

	if not program.nqueries:
		usermsgs.print_warning('No queries in the input file.')
		return 0

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

	if min_sim is None and max_sim is None:
		return 1

	# Call the statistical model checker
	num_sims, qdata = check(program, simulator,
	                        args.seed, args.alpha, args.delta, args.block,
	                        min_sim, max_sim, args.jobs, args.verbose)

	# Print the results on the terminal
	(show_json if args.format == 'json' else show_results)(program, num_sims, qdata)

	# Plot the result of parametric queries if requested
	if args.plot:
		plot_results(program, qdata)

	return 0
