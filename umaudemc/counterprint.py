#
# Counterexample printers
#

import json
import sys
from xml.sax.saxutils import escape

from .common import maude
from .terminal import terminal as tmn

# Colored markers

BAR         = f'{tmn.bold}{tmn.red}|{tmn.reset}'    # A bar in red
ARROW       = f'{tmn.bold}{tmn.red}∨{tmn.reset}'    # An arrow head in red
CYCLE_BAR   = f'{tmn.bold}{tmn.red}| |{tmn.reset}'  # Two bars in red
CYCLE_ARR   = f'{tmn.bold}{tmn.red}| ∨{tmn.reset}'  # A bar and an arrow head in red
CYCLE_END   = f'{tmn.bold}{tmn.red}< ∨{tmn.reset}'  # A loop of arrow heads in red
SOLUTION    = f'{tmn.bold}{tmn.green}X{tmn.reset}'  # An X in green
PREFIX      = f'{tmn.bold}{tmn.cyan}O{tmn.reset}'   # An O in cyan
EDGE_FMT    = f'{tmn.italic}{tmn.cyan}'             # Format for edges
PEND_FMT    = f'{tmn.italic}{tmn.magenta}'          # Format for pending strategies


def print_smc_trans_type(trans):
	"""Describe the type of transition with a word (used in JSON output)"""

	return {
		maude.StrategyRewriteGraph.RULE_APPLICATION	: 'rule',
		maude.StrategyRewriteGraph.OPAQUE_STRATEGY 	: 'opaque',
		maude.StrategyRewriteGraph.SOLUTION		: 'solution'
	}[trans.getType()]

#
# Function that walks through the counterexample calling the printer
#

def is_solution(transition):
	"""Check whether the given transition represents a solution self-loop"""
	return transition.getType() == maude.StrategyRewriteGraph.SOLUTION


def print_counterexample(graph, counter, printer_triple):
	"""Print a model-checking counterexample"""
	lead_in, cycle = counter
	printer, sformat, eformat = printer_triple

	# Is this a trace prefix? (for branching-time properties)
	prefix = (cycle == [])

	if prefix:
		finite = True
		real_leadin_length = len(lead_in) - 1
		real_cycle_length = 0
	else:
		# Is this a finite counterexample trace?
		if graph.strategyControlled:
			finite = is_solution(graph.getTransition(cycle[-1], cycle[0]))
		else:
			finite = graph.getRule(cycle[-1], cycle[0]) is None

		# Reduce the length of the cycle
		real_cycle_length = len(cycle) - 1 if finite else len(cycle)
		real_leadin_length = len(lead_in)

		if (finite and graph.strategyControlled and real_cycle_length > 0 and
			is_solution(graph.getTransition(cycle[real_cycle_length - 1], cycle[real_cycle_length]))):
			real_cycle_length -= 1

	# Begin trace
	printer.begin_trace(finite,
	                    len(lead_in) + real_cycle_length + 1 if finite else len(lead_in),
	                    0 if finite else len(cycle))

	# Lead-in to the cycle
	for i in range(real_leadin_length):
		index = lead_in[i]
		next_index = lead_in[i+1] if i+1 < len(lead_in) else cycle[0]

		if graph.strategyControlled:
			printer.next_step_strat(
				sformat(graph, index),
				graph.getStateStrategy(index),
				eformat(graph, index, next_index),
				sformat(graph, next_index),
				first_index=index,
				second_index=next_index)
		else:
			printer.next_step(
				sformat(graph, index),
				eformat(graph, index, next_index),
				sformat(graph, next_index),
				first_index=index,
				second_index=next_index)

	# Start the loop part
	if not finite:
		printer.start_loop()

	# Cycle
	for i in range(real_cycle_length):
		index = cycle[i]
		next_index = cycle[i+1] if i+1 < len(cycle) else cycle[0]

		if graph.strategyControlled:
			printer.next_step_strat(
				sformat(graph, index),
				graph.getStateStrategy(index),
				eformat(graph, index, next_index),
				sformat(graph, next_index),
				first_index=index,
				second_index=next_index)
		else:
			printer.next_step(
				sformat(graph, index),
				eformat(graph, index, next_index),
				sformat(graph, next_index),
				first_index=index,
				second_index=next_index)

	# Last state for solutions
	if prefix:
		printer.last_state(sformat(graph, lead_in[-1]), index=lead_in[-1], prefix=True)
	elif finite:
		printer.last_state(sformat(graph, cycle[-1]), index=cycle[-1])
	else:
		printer.end_cycle()

#
# Printers should implement the following methods
#
# - begin_trace(whether the trace is finite, size of the path, size of the loop)
# - start_loop() -- the following steps correspond to the loop
# - next_step(origin state, transition, destination) -- a step
# - next_step_strat(origin, pending strategy, transition, destination) -- a step with strategies
# - last_state(state) -- last state of a finite execution
# - end_cycle(state) -- called at the end of the cycle in infinite executions
#
# Method that receive a state get also a first_index, second_index or index
# argument with the index of the state.


class SimplePrinter:
	"""Counterexample printer to the terminal with color and some decorations"""

	def __init__(self, show_strat=False):
		self.show_strat = show_strat

		self.bar = None
		self.arrow = None

	def begin_trace(self, finite, pathSize, loopSize):
		self.bar = BAR
		self.arrow = ARROW

	def start_loop(self):
		self.bar = CYCLE_BAR
		self.arrow = CYCLE_ARR

	def next_step(self, first, transition, second, **kwargs):
		# Print the term at the right of the bar
		for line in str(first).split('\n'):
			print(self.bar, line)
		# Print the transition description at the right of
		# a sequence of arrow heads
		for line in str(transition).split('\n'):
			print(self.arrow, EDGE_FMT, line, tmn.reset)

	def next_step_strat(self, first, pending, transition, second, **kwargs):
		for line in str(first).split('\n'):
			print(self.bar, line)
		# If enabled, prints the next pending strategy after an
		# ampersand following the term
		if self.show_strat:
			for line in str(pending).split('\n'):
				print(self.bar, PEND_FMT, '@', line, tmn.reset)
		for line in str(transition).split('\n'):
			print(self.arrow, EDGE_FMT, line, tmn.reset)

	def last_state(self, state, prefix=False, **kwargs):
		for line in str(state).split('\n'):
			print(PREFIX if prefix else SOLUTION, line)

	def end_cycle(self):
		print(CYCLE_END)


class JSONPrinter:
	"""Counterexample printer to JSON"""

	def __init__(self, ofile=sys.stdout):
		self.root = {}
		self.ofile = ofile
		self.list = None

	def begin_trace(self, finite, pathSize, loopSize):
		self.root['finite'] = finite
		self.root['path'] = []

		self.list = self.root['path']

	def start_loop(self):
		self.root['cycle'] = []
		self.list = self.root['cycle']

	def next_step(self, first, transition, second, **kwargs):
		self.list.append(str(first))
		self.list.append(str(transition))

	def next_step_strat(self, first, pending, transition, second, **kwargs):
		self.list.append({'term': str(first), 'next_strategy': str(pending)})
		self.list.append({'type': print_smc_trans_type(transition), 'value': str(transition)})

	def last_state(self, state, **kwargs):
		self.list.append(str(state))
		print(json.dumps(self.root), file=self.ofile)

	def end_cycle(self):
		print(json.dumps(self.root), file=self.ofile)


class DOTPrinter:
	"""Counterexample printer to GraphViz's DOT"""

	def __init__(self, ofile=sys.stdout):
		self.visited = {}
		self.ofile = ofile

	def begin_trace(self, finite, pathSize, loopSize):
		print('digraph {', file=self.ofile)

	def start_loop(self):
		pass

	def next_step(self, first, transition, second, first_index, second_index, **kwargs):
		print(f'\t{first_index} -> {second_index};', file=self.ofile)
		self.visited.setdefault(first_index, first)

	def next_step_strat(self, first, pending, transition, second, first_index, second_index, **kwargs):
		print(f'\t{first_index} -> {second_index};', file=self.ofile)
		self.visited.setdefault(first_index, first)

	def last_state(self, state, **kwargs):
		self.print_trailer()

	def end_cycle(self):
		self.print_trailer()

	def print_trailer(self):
		for index, value in self.visited.items():
			print(f'\t{index} [label="{escape(str(value))}"]', file=self.ofile)
		print('}', file=self.ofile)


class HTMLPrinter:
	"""Counterexample printer to static HTML (work in progress)"""

	PREAMBLE = '''<!DOCTYPE html>
	<head>
		<meta charset="utf-8" />
		<title>Counterexample</title>
		<link rel="stylesheet" type="text/css" href="counterstyle.css">
	</head>
	<body>
		<h1>Path</h1><table>
	'''

	STATE_LINE = '<tr><td class="state"><pre>{}</pre></td></tr>'
	LAST_STATE_LINE = '<tr><td class="last-state"><pre>{}</pre></td></tr>'

	def __init__(self, ofile=sys.stdout):
		self.root = {}
		self.ofile = ofile

	def begin_trace(self, finite, pathSize, loopSize):
		print(HTMLPrinter.PREAMBLE, file=self.ofile)

	def start_loop(self):
		print('</table><h1>Cycle</h1><table>', file=self.ofile)

	def next_step(self, first, transition, second, **kwargs):
		print(HTMLPrinter.STATE_LINE.format(escape(str(first))), file=self.ofile)
		print('<tr><td class="rule"><pre>{}</pre></td></tr>'.format(escape(str(transition))), file=self.ofile)

	def next_step_strat(self, first, pending, transition, second, **kwargs):
		print(HTMLPrinter.STATE_LINE.format(escape(str(first))), file=self.ofile)
		print('<tr><td class="rule"><pre>{}</pre></td></tr>'.format(escape(str(transition))), file=self.ofile)

	def last_state(self, state, **kwargs):
		print(HTMLPrinter.LAST_STATE_LINE.format(escape(str(state))), file=self.ofile)

	def end_cycle(self):
		print('</table></body></html>', file=self.ofile)
