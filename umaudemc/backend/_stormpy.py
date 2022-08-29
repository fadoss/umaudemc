#
# Storm probabilistic backend for umaudemc (through StormPy)
#

import time

import stormpy

from .prism import make_prism_query, translate_aprop, collect_raw_aprops
from .. import probabilistic as pbs
from .. import usermsgs


class StormBackend:
	"""Storm backend using StormPy"""

	def __init__(self):
		self.satisfies = None
		self.true_term = None
		self.nrRewrites = 0

	def find(self):
		return True

	def _init_aprop(self, graph):
		"""Find the resources required for testing atomic propositions"""

		module = graph.getStateTerm(0).symbol().getModule()
		bool_kind = module.findSort('Bool').kind()

		self.satisfies = module.findSymbol('_|=_', (module.findSort('State').kind(),
		                                            module.findSort('Prop').kind()),
		                                   bool_kind)

		self.true_term = module.findSymbol('true', (), bool_kind).makeTerm(())

	def _check_aprop(self, graph, aprop, stateNr):
		"""Check whether a given atomic proposition holds in a state"""

		t = self.satisfies.makeTerm((graph.getStateTerm(stateNr), aprop))
		self.nrRewrites += t.reduce()
		return t == self.true_term

	def _make_reward(self, graph, evaluator):
		"""Generate a named reward specification for the given evaluator"""

		reward = [0.0] * len(graph)

		for state in graph.states():
			value = evaluator(graph.getStateTerm(state))

			if value:
				reward[state] = value

		return stormpy.SparseRewardModel(optional_state_reward_vector=reward)

	def _build_dtmc(self, builder, graph, ctmc):
		"""Build a DTMC or CTMC from the probabilsitic rewrite graph"""

		first_notvisited, num_states = 0, len(graph)

		for state, children in graph.transitions():
			# Add self-loop to all deadlocked states
			for st in range(first_notvisited, state):
				builder.add_next_value(row=st, column=st, value=1.0)

			first_notvisited = state + 1

			# Normalize if not generating a DTMC
			if not ctmc:
				children = tuple(children)
				total_w = sum(w for w, _ in children)
				children = ((w / total_w, child) for w, child in children)

			for p, child in children:
				builder.add_next_value(row=state, column=child, value=p)

		# Add self-loop to all deadlocked states
		for st in range(first_notvisited, num_states):
			builder.add_next_value(row=st, column=st, value=1.0)

	def _build_mdp(self, builder, graph):
		"""Build an MDP from the probabilsitic rewrite graph"""

		last_visited, row_index, num_states = -1, 0, len(graph)

		for state, children in graph.transitions():
			# Add self-loop to all deadlocked states
			for st in range(last_visited + 1, state):
				builder.new_row_group(row_index)
				builder.add_next_value(row=row_index, column=st, value=1.0)
				row_index += 1

			# Open a new group of choices if a new state
			if state != last_visited:
				builder.new_row_group(row_index)
				last_visited = state

			for p, child in children:
				builder.add_next_value(row=row_index, column=child, value=p)

			row_index += 1

		# Add self-loop to all deadlocked states
		for st in range(last_visited + 1, num_states):
			builder.new_row_group(row_index)
			builder.add_next_value(row=row_index, column=st, value=1.0)
			row_index += 1

	def _build_model(self, graph, aprops, aprop_labels, reward, ctmc):
		"""Build the model programmatically using StormPy"""

		num_states = len(graph)

		# Build an sparse transition matrix
		builder = stormpy.SparseMatrixBuilder(rows=num_states,
		                                      columns=num_states,
		                                      force_dimensions=False,
		                                      has_custom_row_grouping=graph.nondeterminism)

		# For each reachable state, we write its transitions into the matrix
		first_notvisited = 0

		if ctmc or not graph.nondeterminism:
			self._build_dtmc(builder, graph, ctmc)
		else:
			self._build_mdp(builder, graph)

		transition_matrix = builder.build()
		#input(transition_matrix)

		# State labeling with atomic propositions
		state_labeling = stormpy.storage.StateLabeling(num_states)

		for label in aprop_labels:
			state_labeling.add_label(label)

		for k, label in enumerate(aprop_labels):
			bitvec = [s for s in graph.states() if self._check_aprop(graph, aprops[k], s)]
			state_labeling.set_states(label, stormpy.BitVector(num_states, bitvec))

		# Reward models
		if reward is None:
			reward_models = {'': stormpy.SparseRewardModel(optional_state_reward_vector=[1.0] * num_states)}
		else:
			reward_models = {'user': self._make_reward(graph, reward)}

		components = stormpy.SparseModelComponents(transition_matrix, state_labeling, reward_models,
		                                           rate_transitions=ctmc)

		# Decide the kind of probabilistic model
		model_type = (stormpy.storage.SparseCtmc if ctmc else (
			stormpy.storage.SparseMdp if graph.nondeterminism else
			stormpy.storage.SparseDtmc))

		return model_type(components)

	def make_statistics(self, graph, start_time):
		"""Make the model-checking statistics dictionary"""

		return {
			'states': len(graph),
			'rewrites': self.nrRewrites + graph.getNrRewrites(),
			'backend_start_time': start_time,
			'real_states': graph.getNrRealStates()
		}

	def run(self, graph, form, aprops, aprop_labels, cost=False, formula_str=None, reward=None, ctmc=False):
		"""Run the StormPy backend to solve the given problem"""

		# Translate the Maude-parsed or raw formula to a PRISM query
		# (StormPy apparently does not allow building formulas programmatically)
		query = make_prism_query(form, aprops, not graph.nondeterminism,
		                         cost=cost, raw_form=formula_str, reward=reward)

		# Initialize the requirements for evaluating atomic propositions
		self._init_aprop(graph)

		# Build a StormPy model in memory from the probabilistic graph
		model = self._build_model(graph, aprops, aprop_labels, reward, ctmc)

		# Make Storm parse the input query
		try: formulas = stormpy.parse_properties(query)
		except RuntimeError:
			usermsgs.print_error('Error while parsing the probabilistic formula.')
			return None, {}

		# Record the time when the actual backend has run for statistics
		start_time = time.perf_counter_ns()

		try: results = [stormpy.model_checking(model, form) for form in formulas]
		except RuntimeError:
			usermsgs.print_error('Error while model checking with StormPy.')
			return None, {}

		# Results might be null even if no RuntimeError has been raised
		if None in results:
			usermsgs.print_error(f'Error while model checking with StormPy.')
			return None, {}

		# Prepare the statistics dictionary
		stats = self.make_statistics(graph, start_time) if results is not None else None

		# Construct a QuantitativeResult
		if isinstance(results[0].at(0), bool):
			result = pbs.QuantitativeResult.make_boolean(results[0].at(0))
		else:
			result = pbs.QuantitativeResult.make_number(results[0].at(0))

			if len(results) > 1:
				result = pbs.QuantitativeResult.make_range(result, results[1].at(0))

		return result, stats

	def check(self, graph=None, module=None, formula=None, formula_str=None,
	          aprops=None, cost=False, reward=None, ctmc=False, **_):
		"""Solves a model-checking problem with Storm"""

		# Extract the atomic proposition term from the raw formulae
		if formula is None:
			aprops = set()
			collect_raw_aprops(formula_str, aprops)

		# Parse the atomic proposition to be evaluated on the states
		aprop_terms = [module.parseTerm(prop) for prop in aprops]
		aprop_labels = [translate_aprop(prop) for prop in aprops]

		holds, stats = self.run(graph, formula, aprop_terms, aprop_labels, cost=cost,
		                        formula_str=formula_str, reward=reward, ctmc=ctmc)

		return holds, stats

	def state_analysis(self, *args, **kwargs):
		"""Steady and transient state analysis"""

		# StormPy does not allow calculting state probabilities, as far as we know, so
		# we delegate on the command-line interface to Storm
		from ._storm import StormBackend as StormCmdLineBackend

		cmdline = StormCmdLineBackend()

		if not cmdline.find():
			usermsgs.print_error('StormPy does not support state-probability analyses and the Storm command '
			                     'cannot be found (STORM_PATH may not be set appropriately).')
			return None, {}

		return cmdline.state_analysis(*args, **kwargs)
