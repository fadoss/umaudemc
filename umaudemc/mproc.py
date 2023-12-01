#
# Separate Maude processes within Python
#
# When the Python interpreter loads the maude package a
# session of Maude is started and cannot be restarted
# or stopped if something goes wrong.
#

import multiprocessing as mp
import os


class MaudeProcess:
	"""
	Server for the Python instance running the Maude interpreter.

	It communicates with the client using pipes where the method to be
	called and its arguments are received and the results are sent
	as a dictionary.
	"""

	DEFAULT_BACKENDS = 'maude,ltsmin,pymc,nusmv,spot,spin,builtin'
	DEFAULT_PBACKENDS = 'prism,storm'

	def __init__(self, backends=DEFAULT_BACKENDS):
		from .common import maude, InitialData
		from . import wrappers  # for getNextStates and strategyControlled
		from .backends import get_backends, backend_for
		from .formulae import Parser
		maude.init(advise=False)

		self.graph = None
		self.names = {
			maude.Module.SYSTEM_MODULE: 'mod',
			maude.Module.STRATEGY_MODULE: 'smod',
			maude.Module.SYSTEM_THEORY: 'th',
			maude.Module.STRATEGY_THEORY: 'sth',
		}

		# The Maude module is saved as an attribute since it cannot be
		# a global variable without forcing the client to load it
		self.maude = maude
		self.backends, unsupported = get_backends(backends)
		self.backend_for = backend_for
		self.Parser = Parser
		self.InitialData = InitialData

		# The probabilistic ingredients are lazily imported
		self.ProbParser = None
		self.get_probabilistic_graph = None
		self.RewardEvaluator = None
		self.prob_backend = None

		# Dictionary of jobs where model-checking task are stored
		self.filename = None
		self.jobs = {}
		self.counter = 0

	def load(self, filename):
		self.filename = filename
		return self.maude.load(filename)

	def getModules(self):
		modlist = []

		# Filters functional modules and theories
		# (without interest for model checking)
		for mod in self.maude.getModules():
			if (mod.type != self.maude.Module.FUNCTIONAL_MODULE and
			    mod.type != self.maude.Module.FUNCTIONAL_THEORY):
				modlist.append((mod.name, self.names[mod.type]))

		return modlist

	def getModuleInfo(self, name):
		modinfo = {'name': name, 'valid': False}
		mod = self.maude.getModule(name)

		if mod is None:
			return None

		modinfo['type'] = self.names[mod.getModuleType()]

		# Quickly check whether the module is valid for model checking
		state_sort = mod.findSort('State')
		prop_sort = mod.findSort('Prop')

		if state_sort is None or prop_sort is None:
			return modinfo

		modinfo['valid'] = True

		# Obtain the state sorts (subsorts of State)
		modinfo['state'] = [str(sort) for sort in state_sort.getSubsorts()]

		# Obtain the list of atomic propositions with their signatures
		atomic_props = []

		for symbol in mod.getSymbols():
			if symbol.getRangeSort() <= prop_sort:
				prop_signature = [str(symbol)]
				opdecls = symbol.getOpDeclarations()
				first_decl = opdecls[0]
				decl_domain = first_decl.getDomainAndRange()

				for i in range(symbol.arity()):
					prop_signature.append(str(decl_domain[i]))

				atomic_props.append(prop_signature)

		modinfo['prop'] = atomic_props

		# Obtain the list of strategies with their signatures
		strats = []

		for strat in mod.getStrategies():
			if strat.getSubjectSort() <= state_sort:
				strat_signature = [strat.getName()]

				for arg in strat.getDomain():
					strat_signature.append(str(arg))

				strats.append(strat_signature)

		modinfo['strat'] = strats

		return modinfo

	def get_result(self, index):
		task = self.jobs.pop(index, None)

		if task is None:
			return None

		# Executes the actual model checking
		result, stats = task()

		if result is None or 'counterexample' not in stats:
			return {
				'hasCounterexample': False,
				**self._make_result(result)
			}
		else:
			lead_in, cycle = stats['counterexample']
			graph = stats['graph']

			states_involved = set(lead_in).union(set(cycle))

			return {
				'result': result,
				'rtype': 'b',
				'hasCounterexample': True,
				'leadIn': list(lead_in),
				'cycle': list(cycle),
				'states': {index: self._state_summary(graph, index) for index in states_involved},
			}

	def modelCheck(self, data):
		"""Model check any supported property"""

		# Selects the module
		mod = self.maude.getModule(data['module'])

		if mod is None:
			return {'ok': False, 'cause': 'module'}

		# Parses the initial term
		state_term = mod.findSort('State')

		if state_term is None:
			return {'ok': False, 'cause': 'state'}

		initial = mod.parseTerm(data['initial'], state_term.kind())

		if initial is None:
			return {'ok': False, 'cause': 'term'}

		# Parses the strategy expression
		if data['strat']:
			strategy = mod.parseStrategy(data['strat'])

			if strategy is None:
				return {'ok': False, 'cause': 'strat'}
		else:
			strategy = None

		# Opaque strategies
		opaques = data.get('opaques', [])

		# Make the (qualitative or quantitative) model-checking task
		make_task = self._make_pcheck_task if data.get('passign') else self._make_check_task
		task, other = make_task(data, mod, initial, strategy, opaques)

		if task is None:
			return other

		# Add this model-checking job to the table
		self.jobs[self.counter] = task
		self.counter += 1

		return {
			'ok'	: True,
			'logic'	: other,
			'ref'	: self.counter - 1
		}

	def _make_check_task(self, data, mod, initial, strategy, opaques):
		"""Make a qualitative model-checking task"""

		# Parse the qualitative formula
		parser = self.Parser()
		parser.set_module(mod)
		formula, logic = parser.parse(data['formula'], opaques=opaques)

		if formula is None or logic == 'invalid':
			return None, {'ok': False, 'cause': 'formula'}

		# Find a backend for the given formula
		name, handle = self.backend_for(self.backends, logic)

		if name is None:
			return None, {'ok': False, 'logic': logic, 'cause': 'nobackend'}

		# Save the model-checking task as a lambda
		task = lambda: handle.check(module=mod,
		                            module_str=data['module'],
		                            term=initial,
		                            term_str=data['initial'],
		                            strategy=strategy,
		                            strategy_str=data['strat'],
		                            opaque=opaques,
		                            full_matchrew=False,
		                            formula=formula,
		                            formula_str=data['formula'],
		                            logic=logic,
		                            labels=parser.labels,
		                            filename=self.filename,
		                            get_graph=True)

		return task, logic

	def _make_pcheck_task(self, data, mod, initial, strategy, opaques):
		"""Make a quantitative model-checking task"""

		# Lazily import the probabilistic components
		if not self.ProbParser:
			from .formulae import ProbParser
			from .backend import prism, storm
			from .probabilistic import get_probabilistic_graph, RewardEvaluator

			self.ProbParser = ProbParser
			self.get_probabilistic_graph = get_probabilistic_graph
			self.RewardEvaluator = RewardEvaluator

			# Find a probabilistic backend
			backend_list = os.getenv('UMAUDEMC_PBACKEND') or \
			               self.DEFAULT_PBACKENDS

			for name in backend_list.split(','):
				if name == 'prism':
					backend = prism.PRISMBackend()
				elif name == 'storm':
					backend = storm.StormBackend()
				else:
					continue

				if backend.find():
					self.prob_backend = backend

		if self.prob_backend is None:
			return None, {'ok': False, 'logic': 'prob', 'cause': 'nopbackend'}

		# Parse the quantitative formula
		parser = self.ProbParser()
		parser.set_module(mod)
		formula, aprops = parser.parse(data['formula'])

		if formula is None:
			return None, {'ok': False, 'cause': 'formula'}

		# Generate the probabilistic graph (using passign)
		cdata = self.InitialData()

		cdata.module = mod
		cdata.strategy = strategy
		cdata.term = initial
		cdata.opaque = opaques
		cdata.full_matchrew = False

		graph = self.get_probabilistic_graph(cdata, data['passign'])

		if graph is None:
			return None, {'ok': False, 'logic': 'prob', 'cause': 'passign'}

		# Parse the reward (if any)
		reward, reward_term = None, data['reward']

		if reward_term and reward_term != 'steps':
			reward_term = mod.parseTerm(reward_term)

			if reward_term:
				reward = self.RewardEvaluator.new(reward_term, initial.getSort().kind())

			if not reward_term:
				return None, {'ok': False, 'logic': 'prob', 'cause': 'reward'}

		# Solve the given problem
		task = lambda: self.prob_backend.check(module=mod,
		                                       formula=formula,
		                                       formula_str=data['formula'],
		                                       logic='CTL',
		                                       aprops=aprops,
		                                       cost=reward is None and reward_term == 'steps',
		                                       reward=reward,
		                                       graph=graph)

		return task, 'prob'

	def _make_result(self, result):
		"""Transform the result for being sent to the client"""

		if result is None:
			return {'result': None}

		if isinstance(result, bool):
			return {'result': result, 'rtype': 'b'}

		# result is a QuantitativeResult
		return {'result': result.value, 'rtype': 'bnr'[result.rtype - 1]}

	def _state_summary(self, graph, index):
		"""Generate a summary of a state for its graphical representation"""

		return {
			'solution'	: graph.strategyControlled and graph.isSolutionState(index),
			'term'		: str(graph.getStateTerm(index)),
			'strategy'	: str(graph.getStateStrategy(index)) if graph.strategyControlled else '',
			'successors'	: [self._transition_summary(graph, index, next_state)
			                   for next_state in graph.getNextStates(index)]
		}

	def _transition_summary(self, graph, origin, dest, full=False):
		"""Generate a summary of a transition for its graphical representation"""

		if graph.strategyControlled:
			trans = graph.getTransition(origin, dest)

			ttype = {
				self.maude.StrategyRewriteGraph.RULE_APPLICATION: 1,
				self.maude.StrategyRewriteGraph.OPAQUE_STRATEGY: 2,
				self.maude.StrategyRewriteGraph.SOLUTION: 0
			}[trans.getType()]

			if ttype == 1:
				label = trans.getRule()
				label = str(label) if full else label.getLabel()
			elif ttype == 2:
				label = trans.getStrategy()
				label = str(label) if full else label.getName()
			else:
				label = ''
		else:
			ttype = 1
			label = graph.getRule(origin, dest)
			label = str(label) if full else label.getLabel()

		return {'target': dest, 'type': ttype, 'label': label}

	@staticmethod
	def run(pipe):
		maudp = MaudeProcess()
		msg = pipe.recv()

		while msg:
			# Call the given method and send the result
			func, *args = msg
			pipe.send(func(maudp, *args))

			msg = pipe.recv()


class MaudeRemote:
	"""Remote that communicates with the Maude instance"""

	def __init__(self):
		self.pipe, child_pipe = mp.Pipe()
		self.p = mp.Process(target=MaudeProcess.run, args=(child_pipe, ))
		self.p.start()

	def load(self, filename):
		self.pipe.send((MaudeProcess.load, filename))
		return self.pipe.recv()

	def getModules(self):
		self.pipe.send((MaudeProcess.getModules,))
		return self.pipe.recv()

	def getModuleInfo(self, name):
		self.pipe.send((MaudeProcess.getModuleInfo, name))
		return self.pipe.recv()

	def modelCheck(self, data):
		self.pipe.send((MaudeProcess.modelCheck, data))
		return self.pipe.recv()

	def get_result(self, data):
		self.pipe.send((MaudeProcess.get_result, data))
		return self.pipe.recv()

	def shutdown(self):
		self.pipe.send(())
		self.p.join(timeout=2)
		self.p.terminate()
		self.p = None
		self.pipe = None

	def forced_shutdown(self):
		self.p.kill()
		self.p = None
		self.pipe = None
