#
# Separate Maude processes within Python
#
# When the Python interpreter loads the maude package a
# a session of Maude is started and cannot be restarted
# or stopped if something goes wrong.
#

import multiprocessing as mp


class MaudeProcess:
	"""
	Server for the Python instance running the Maude interpreter.

	It communicates with the client using pipes where the method to be
	called and its arguments are receives and the results are sent
	as a dictionary.
	"""

	DEFAULT_BACKENDS = 'maude,ltsmin,pymc,nusmv,spot,builtin'

	def __init__(self, backends=DEFAULT_BACKENDS):
		from .common import maude
		from . import wrappers
		from .backends import get_backends, backend_for
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
		actual_state_sort = []
		for sort in state_sort.getSubsorts():
			actual_state_sort.append(str(sort))

		modinfo['state'] = actual_state_sort

		# Obtain the list of atomic propositions with their signatures
		atomic_props = []

		for symbol in mod.getSymbols():
			if symbol.getRangeSort() <= prop_sort:
				prop_signature = [str(symbol)]
				opdecls = symbol.getOpDeclarations()
				first_decl = opdecls[0]
				decl_domain = first_decl.getDomainAndRange()

				for i in range(0, symbol.arity()):
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
		holds, stats = task()

		if 'counterexample' in stats:
			lead_in, cycle = stats['counterexample']
			graph = stats['graph']

			states_involved = set(lead_in).union(set(cycle))

			return {
				'hasCounterexample': True,
				'holds': holds,
				'leadIn': list(lead_in),
				'cycle': list(cycle),
				'states': {index: self._state_summary(graph, index) for index in states_involved}
			}
		else:
			return {
				'holds': holds,
				'hasCounterexample': False
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

		# Parse the LTL formula
		from .formulae import Parser

		# Opaque strategies
		opaques = data.get('opaques', [])

		parser = Parser()
		parser.set_module(mod)
		formula, logic = parser.parse(data['formula'], opaques=opaques)

		if formula is None or logic == 'invalid':
			return {'ok': False, 'cause': 'formula'}

		# Find a backend for the given formula
		name, handle = self.backend_for(self.backends, logic)

		if name is None:
			return {'ok': False, 'logic': logic, 'cause': 'nobackend'}

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

		# Add this model-checking job to the table
		self.jobs[self.counter] = task
		self.counter += 1

		return {
			'ok'	: True,
			'logic'	: logic,
			'ref'	: self.counter - 1
		}

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
			func, args = msg[0], msg[1:]
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
