#
# Implementation of the strategy language in Python
# (for the analytic generation of probabilistic models)
#

import itertools
import random

import maude

# Deep hashing and comparison of substitutions
maude.Substitution.__hash__ = lambda s: hash(tuple(s))
maude.Substitution.__eq__ = lambda s1, s2: tuple(s1) == tuple(s2)


class Instruction:
	"""Instruction of the strategy language machine"""

	POP = 0  # no arguments
	JUMP = 1  # sequence of addresses
	STRAT = 2  # strategy (at the object level, for potential optimizations only)
	RLAPP = 3  # (label, intial substitution, whether it is applied on top)
	TEST = 4  # (match type, pattern, condition)
	CHOICE = 5  # sequence of pairs of term (weight) and address
	CALL = 6  # (call term, index of a named strategy within the program, whether the call is tail)
	SUBSEARCH = 7  # address of the instruction to execute when the search fails
	NOFAIL = 8  # no arguments
	MATCHREW = 9  # (maximum depth, pattern, condition, list of subterm variables)
	NEXTSUBTERM = 10  # (whether this is the last subterm, subterm variable)
	RWCSTART = 11  # list of (pattern, condition, starting pattern)
	RWCNEXT = 12  # list of (end pattern, condition, starting pattern/right-hand side)
	NOTIFY = 13  # list of lists of variables of the nested pending matchrew subterms
	SAMPLE = 14  # (variable, distribution callable, its arguments)
	WMATCHREW = 15  # like MATCHREW + (weight)
	ONE = 16  # no arguments
	CHECKPOINT = 17  # no arguments

	NAMES = ['POP', 'JUMP', 'STRAT', 'RLAPP', 'TEST', 'CHOICE', 'CALL', 'SUBSEARCH',
	         'NOFAIL', 'MATCHREW', 'NEXTSUBTERM', 'RWCSTART', 'RWCNEXT', 'NOTIFY',
	         'SAMPLE', 'WMATCHREW', 'ONE', 'CHECKPOINT']

	def __init__(self, itype, extra=None):
		self.type = itype
		self.extra = extra

	def __repr__(self):
		return f'Instruction({self.NAMES[self.type]}, {self.extra})'


class StratProgram:
	"""Compiled strategy program"""

	def __init__(self):
		self.inst = []  # Instructions
		self.defs = {}  # Strategy definition: pattern, conditions, and addresses (keys are names, just for debugging)

	def append(self, itype, extra=None):
		"""Append an instruction to the program"""

		self.inst.append(Instruction(itype, extra))
		return self.inst[-1]

	def pop(self):
		"""Remove the last instruction of the program"""
		return self.inst.pop()

	def __len__(self):
		return len(self.inst)

	@property
	def pc(self):
		"""Current program counter (a synonym for the length)"""
		return len(self.inst)

	def __getitem__(self, k):
		"""Get the instruction of the program at the given index"""
		return self.inst[k]

	def __repr__(self):
		return f'StratProgram({self.inst}, {self.defs})'

	def dump(self):
		"""Dump the program in human-readable form (for debugging)"""

		# Lines where the body of a strategy definition starts
		entry_points = {line: name for name, defs in self.defs for _, _, line in defs}

		for k, inst in enumerate(self.inst):
			if k in entry_points:
				print(f'\x1b[36m{entry_points[k]}:\x1b[0m')

			args = '' if inst.extra is None else inst.extra

			if isinstance(args, tuple):
				args = ' '.join(map(str, args))
			else:
				args = '' if inst.extra is None else inst.extra

			if inst.type == Instruction.CALL:
				call, strat_nr, tail = inst.extra
				print(f'{k:<5}{"CALL":18}{strat_nr} ({" ".join(map(str, call.arguments()))}) '
				      f"{' (tail)' if tail else ''}"
				      f'\t\x1b[32m# {self.defs[strat_nr][0]}\x1b[0m')

			else:
				print(f'{k:<5}{Instruction.NAMES[inst.type]:18}{args}')


def instantiate_condition(condition, subs):
	"""Instantiate a condition with a given substitution"""

	return [
		maude.SortTestCondition(subs.instantiate(cf.getLhs()), cf.getSort())
		if isinstance(cf, maude.SortTestCondition)
		else type(cf)(subs.instantiate(cf.getLhs()), subs.instantiate(cf.getRhs()))
		for cf in condition
	]


def merge_substitutions(sb1, sb2):
	"""Merge two substitutions"""

	if sb1 is None or len(sb1) == 0:
		return sb2 if isinstance(sb2, maude.Substitution) else maude.Substitution(sb2)

	if sb2 is None or len(sb2) == 0:
		return sb1

	return maude.Substitution({**dict(sb1), **dict(sb2)})


def bernoulli_distrib(p):
	"""Bernoulli distribution"""

	return 1.0 if random.random() < p else 0.0


def find_cycles(initial, next_fn):
	"""Find cycles in the graph and return the points where to cut them"""

	cut_points = set()

	# This is a depth-first search using an explicit stack (instead of recursion, due to
	# the limitations of Python). The stack holds a vertex of the graph and an iterator
	# over its children, which is initially obtained with the parameter next_fn.
	dfs_stack = [(initial, next_fn(initial))]
	visited, in_stack = {initial}, {initial}

	while dfs_stack:
		state, it = dfs_stack[-1]
		next_state = next(it, None)

		if next_state is not None:
			# Continue the DFS from this new vertex
			if next_state not in visited:
				visited.add(next_state)
				in_stack.add(next_state)
				dfs_stack.append((next_state, next_fn(next_state)))

			# A cycle has been found, mark the state as a cut point
			elif next_state in in_stack:
				cut_points.add(next_state)

		# We have finished with the children of state (and with its subtree)
		else:
			dfs_stack.pop()
			in_stack.remove(state)

	return cut_points, visited


class ListWalker:
	"""Helper class to walk through a sequence (useful idiom in the following)"""

	def __init__(self, lst):
		self.lst = lst
		self.index = 0

	def reset(self):
		self.index = 0

	def __contains__(self, value):
		return value in self.lst

	def peek_equal(self, value):
		"""Check whether the next number of the sequence is the given one without advancing it"""

		return self.index < len(self.lst) and self.lst[self.index] == value

	def next_equal(self, value):
		"""Check whether the next number of the sequence is the given one"""

		if self.peek_equal(value):
			self.index += 1
			return True

		return False

	def next_between(self, lbound, ubound):
		"""Check whether the next number of the sequence is within the bounds"""

		if self.index < len(self.lst) and lbound <= self.lst[self.index] < ubound:
			self.index += 1
			return True

		return False


class StratCompiler:
	"""Compiler of strategy expressions"""

	def __init__(self, m, ml, use_notify=False, ignore_one=False):
		# Metalevel module
		self.ml = ml
		# Object-level module where the strategy lives
		self.m = m

		# Whether notify instructions are generated
		self.use_notify = use_notify
		# Whether the one combinator is ignored or not
		self.use_one = not ignore_one

		# Table of strategy combinators
		self.op_table = {}

		# Relevant kinds
		st_kind = ml.findSort('Strategy').kind()
		term_kind = ml.findSort('Term').kind()
		subs_kind = ml.findSort('Substitution').kind()
		using_pair_kind = ml.findSort('UsingPair').kind()
		condition_kind = ml.findSort('Condition').kind()

		# Kind of strategy metarepresentations
		self.st_kind = st_kind

		# Strategy combinators
		# (their symbols are found and bound to handlers)
		self.bind_op('idle', 'idle')
		self.bind_op('fail', 'fail')
		self.bind_op('all', 'all')
		self.bind_op('top', 'top', st_kind)
		self.bind_op('rlapp', '_[_]{_}', term_kind, subs_kind, st_kind)
		self.bind_op('disj', '_|_', st_kind, st_kind)
		self.bind_op('conc', '_;_', st_kind, st_kind)
		self.bind_op('orelse', '_or-else_', st_kind, st_kind)
		self.bind_op('ne_iter', '_+', st_kind)
		self.bind_op('cond', '_?_:_', st_kind, st_kind, st_kind)
		for prefix in ('', 'x', 'a'):
			self.bind_op(f'{prefix}match', f'{prefix}match_s.t._', term_kind, condition_kind)
			self.bind_op(f'{prefix}mrew', f'{prefix}matchrew_s.t._by_', term_kind, condition_kind, using_pair_kind)
		self.bind_op('call', '_[[_]]', term_kind, term_kind)
		self.bind_op('one', 'one', st_kind)

		self.bind_op('iter', '_*', st_kind)
		self.bind_op('norm', '_!', st_kind)
		self.bind_op('stry', 'try', st_kind)
		self.bind_op('stest', 'test', st_kind)
		self.bind_op('snot', 'not', st_kind)

		# Probabilistic strategy combinators
		choice_map_sort = ml.findSort('ChoiceMap')

		if choice_map_sort:
			self.bind_op('choice', 'choice', choice_map_sort.kind())
			self.bind_op('sample', 'sample_:=_in_', term_kind, term_kind, st_kind)
			for prefix in ('', 'x', 'a'):
				self.bind_op(f'{prefix}mrew_ww', f'{prefix}matchrew_s.t._with`weight_by_',
				             term_kind, condition_kind, term_kind, using_pair_kind)

		# Other auxiliary constants and symbols
		self.idle_const = ml.findSymbol('idle', (), st_kind).makeTerm(())
		self.fail_const = ml.findSymbol('fail', (), st_kind).makeTerm(())
		self.all_const = ml.findSymbol('all', (), st_kind).makeTerm(())

		self.empty_st_list = ml.findSymbol('empty', (), st_kind).makeTerm(())
		self.st_list_symbol = ml.findSymbol('_,_', (st_kind, st_kind), st_kind)
		self.empty_term_list = ml.findSymbol('empty', (), term_kind).makeTerm(())
		self.term_list_symbol = ml.findSymbol('_,_', (term_kind, term_kind), term_kind)
		self.using_symbol = ml.findSymbol('_using_', (term_kind, st_kind), ml.findSort('UsingPair').kind())
		self.empty_subs = ml.findSymbol('none', (), subs_kind).makeTerm(())
		self.subs_list_symbol = ml.findSymbol('_;_', (subs_kind, subs_kind), subs_kind)

		# Conditions
		self.nilc = ml.findSymbol('nil', (), condition_kind).makeTerm(())
		self.and_symbol = ml.findSymbol('_/\\_', (condition_kind, condition_kind), condition_kind)
		self.eqc_symbol = ml.findSymbol('_=_', (term_kind, term_kind), condition_kind)
		self.mtc_symbol = ml.findSymbol('_:=_', (term_kind, term_kind), condition_kind)
		self.stc_symbol = ml.findSymbol('_:_', (term_kind, term_kind), condition_kind)

		# Program counters of the generated strategy calls
		self.calls = []

	def bind_op(self, func, name, *args):
		"""Bind a strategy combinator to a handler"""

		symbol = self.ml.findSymbol(name, args, self.st_kind)
		self.op_table[symbol] = func

	@staticmethod
	def get_list(term, empty, join):
		"""Convert a list-like term to a Python list"""

		if term == empty:
			return []

		if term.symbol() != join:
			return [term]

		return list(term.arguments())

	def substitution2dict(self, substitution):
		"""Convert a metalevel substitution to a dictionary"""

		substitution_list = self.get_list(substitution, self.empty_subs, self.subs_list_symbol)

		return {self.m.downTerm(variable): self.m.downTerm(value)
		        for variable, value in map(maude.Term.arguments, substitution_list)}

	@staticmethod
	def nrewc(rule):
		"""Number of rewriting conditions"""

		return sum(1 for cf in rule.getCondition() if isinstance(cf, maude.RewriteCondition))

	def idle(self, s, p, tail):
		# No code is generated for idle
		pass

	def fail(self, s, p, tail):
		# A failure is a jump instruction without alternatives
		p.append(Instruction.JUMP, extra=())

	def all(self, s, p, tail):
		p.append(Instruction.RLAPP, extra=(None, None, False))

	def rlapp(self, s, p, tail):
		self.make_rlapp(s, p, on_top=False)

	def top(self, s, p, tail):
		arg, = s.arguments()

		if arg == self.all_const:
			p.append(Instruction.RLAPP, extra=(None, None, True))

		else:
			self.make_rlapp(arg, p, on_top=True)

	def make_rlapp(self, s, p, on_top):
		"""Generate code for a rule application"""

		label, substitution, substrats = s.arguments()
		label = str(label)[1:]
		substitution = self.substitution2dict(substitution)

		if substrats == self.empty_st_list:
			p.append(Instruction.RLAPP, extra=(label, substitution if substitution else None, on_top))
		else:
			self.make_rewc(p, label, substitution, substrats, top=on_top)

	def make_rewc(self, p, label, substitution, substrats, top=False):
		"""Generate code for applying a rule with rewriting conditions"""

		substrats = self.get_list(substrats, self.empty_st_list, self.st_list_symbol)

		# Find all applicable rules (with the same label and number of rewriting conditions)
		applicable_rules = [rl for rl in self.m.getRules() if rl.getLabel() == label
		                    and self.nrewc(rl) == len(substrats)]

		# If no rule is applicable, just fail (this might be a compile-time error)
		if not applicable_rules:
			p.append(Instruction.JUMP, extra=())
			return

		# For each alternative applicable rule, we split the condition by rewriting
		# fragments and distribute it among different RWCSTART and RWCNEXT instructions
		alternatives = []

		for rl in applicable_rules:
			lhs = rl.getLhs()

			# split_condition always have even length and contains tuples with equational
			# condition fragments followed by a rewriting condition fragment
			split_condition, current_condition = [], []

			for cf in rl.getCondition():
				if isinstance(cf, maude.RewriteCondition):
					split_condition.append(tuple(current_condition))
					split_condition.append(cf)
					current_condition.clear()
				else:
					current_condition.append(cf)

			split_condition.append(tuple(current_condition))
			alternatives.append((lhs, split_condition))

		# Rule left-hand side, first equational condition segment, and first rewriting
		# condition's left-hand side for the initial RWCSTART instruction
		first_alternative = tuple((lhs, c[0], c[1].getLhs()) for lhs, c in alternatives)

		p.append(Instruction.RWCSTART, (top, substitution, first_alternative))

		# Generate RWCNEXT to transit from one rewriting fragment to the next one
		# (the tuple contains the right-hand side of the previous search, the next
		# equational condition segment, and the left-hand side of the next search)
		for k, subst in enumerate(substrats[:-1]):
			self.generate(subst, p, False)
			p.append(Instruction.RWCNEXT, (
				False,
				tuple((c[2 * k + 1].getRhs(), c[2 * k + 2], c[2 * k + 3].getLhs())
				      for _, c in alternatives)
			))

		# Generate the final rewriting fragment code and the instruction doing the rewrite
		self.generate(substrats[-1], p, False)

		p.append(Instruction.RWCNEXT, (
			True,
			tuple((c[-2].getRhs(), c[-1], applicable_rules[k].getRhs())
			      for k, (_, c) in enumerate(alternatives))
		))

	def conc(self, s, p, tail):
		args = list(s.arguments())

		# Concatenation is just the concatenation of the generated code
		for child in args[:-1]:
			self.generate(child, p, False)

		self.generate(args[-1], p, tail)

	def disj(self, s, p, tail):
		# Disjunctions are implemented with an initial jump instruction with
		# as many addresses as branches (there are accumulated in addrs below)
		addrs, jumps = [], []
		initial_jump = p.append(Instruction.JUMP)

		for child in s.arguments():
			addrs.append(p.pc)
			self.generate(child, p, tail)

			# Jump over the other branches
			jumps.append(p.append(Instruction.JUMP))

		# The last branch does not need to jump
		jumps.pop()
		p.pop()

		# Set the addresses of the initial jump and the branch jumps
		initial_jump.extra = tuple(addrs)
		for jump in jumps:
			jump.extra = (p.pc,)

	def ne_iter(self, s, p, tail):
		initial_pc = p.pc

		# The non-empty iteration executes the body and then jumps
		# to the next instruction or to the body again
		self.generate(next(s.arguments()), p, False)

		p.append(Instruction.JUMP, (p.pc + 1, initial_pc))

	def iter(self, s, p, tail):
		# The empty iteration does the same with an additional initial jump
		initial_jump = p.append(Instruction.JUMP)
		initial_pc = p.pc

		self.generate(next(s.arguments()), p, False)

		initial_jump.extra = (p.pc + 1, initial_pc)
		p.append(Instruction.JUMP, (p.pc + 1, initial_pc))

	def cond(self, s, p, tail):
		condition, true_branch, false_branch = s.arguments()

		self.make_conditional(condition, true_branch, false_branch, p, tail)

	def make_conditional(self, condition, true_branch, false_branch, p, tail):
		"""Generate code for a conditional expression"""

		# The condition's code is surrounded by a SUBSEARCH and a NOFAIL instruction.
		# SUBSEARCH is supposed to record the initial term and the address of the
		# negative branch to jump there in case of a failure.
		subsearch = p.append(Instruction.SUBSEARCH)
		self.generate(condition, p, False)

		# NOFAIL disables the execution of the negative branch (because a
		# solution for the condition has been found)
		p.append(Instruction.NOFAIL)

		self.generate(true_branch, p, tail)

		# A jump is needed after the positive branch to skip the negative one
		if false_branch != self.idle_const and true_branch != self.fail_const:
			jump = p.append(Instruction.JUMP)
		else:
			jump = None

		# The address of the negative branch is stored in SUBSEARCH
		subsearch.extra = p.pc
		self.generate(false_branch, p, tail)

		if jump:
			jump.extra = (p.pc, )

	def orelse(self, s, p, tail):
		first, second = s.arguments()

		# A simplification of the conditional (without positive branch)
		self.make_conditional(first, self.idle_const, second, p, tail)

	def match(self, s, p, tail):
		self.test(s, p, -1)

	def xmatch(self, s, p, tail):
		self.test(s, p, 0)

	def amatch(self, s, p, tail):
		self.test(s, p, maude.UNBOUNDED)

	def test(self, s, p, mtype):
		pattern, condition = s.arguments()

		pattern = self.m.downTerm(pattern)
		condition = self.downCondition(condition)

		p.append(Instruction.TEST, extra=(mtype, pattern, condition))

	def mrew(self, s, p, tail):
		self.matchrew(s.arguments(), p, -1, tail)

	def xmrew(self, s, p, tail):
		self.matchrew(s.arguments(), p, 0, tail)

	def amrew(self, s, p, tail):
		self.matchrew(s.arguments(), p, maude.UNBOUNDED, tail)

	def mrew_ww(self, s, p, tail):
		self.matchrew_weight(s, p, -1, tail)

	def xmrew_ww(self, s, p, tail):
		self.matchrew_weight(s, p, 0, tail)

	def amrew_ww(self, s, p, tail):
		self.matchrew_weight(s, p, maude.UNBOUNDED, tail)

	def matchrew(self, sargs, p, mtype, tail):
		"""Generate code for a matchrew"""

		pattern, condition, clauses = sargs

		if clauses.symbol() == self.using_symbol:
			clauses = [clauses]
		else:
			clauses = list(clauses.arguments())

		# Main pattern
		pattern = self.m.downTerm(pattern)

		initial_inst = p.append(Instruction.MATCHREW)
		last_subterm = len(clauses) - 1

		# Variable selecting the subterms to be rewritten
		variables = [self.m.downTerm(next(clause.arguments())) for clause in clauses]

		# Generate code for each branch of the matchrew separated by NEXTSUBTERM
		for k, clause in enumerate(clauses):
			_, bs = clause.arguments()

			self.generate(bs, p, False)

			if k != last_subterm:
				p.append(Instruction.NEXTSUBTERM, extra=(False, variables[k]))

		p.append(Instruction.NEXTSUBTERM, extra=(True, variables[-1]))

		condition = self.downCondition(condition)
		initial_inst.extra = (mtype, pattern, condition, tuple(variables))

		return initial_inst

	def matchrew_weight(self, s, p, mtype, tail):
		"""Generate code for a matchrew with weight"""

		pattern, condition, weight, clauses = s.arguments()

		# Generate the same code as a classical matchrew
		initial_instr = self.matchrew((pattern, condition, clauses), p, mtype, tail)

		# Change the instruction to weighted version
		initial_instr.type = Instruction.WMATCHREW
		initial_instr.extra += (self.m.downTerm(weight), )

	def downCondition(self, condition):
		"""Process a metalevel condition"""

		return list(map(self.downConditionFragment,
		                self.get_list(condition, self.nilc, self.and_symbol)))

	def downConditionFragment(self, fragment):
		"""Process a metalevel condition fragment"""

		symbol = fragment.symbol()
		lhs, rhs = fragment.arguments()
		lhs = self.m.downTerm(lhs)

		if symbol == self.eqc_symbol:
			return maude.EqualityCondition(lhs, self.m.downTerm(rhs))

		if symbol == self.mtc_symbol:
			return maude.AssignmentCondition(lhs, self.m.downTerm(rhs))

		if symbol == self.stc_symbol:
			return maude.SortTestCondition(lhs, self.m.findSort(str(rhs)[1:]))

	def norm(self, s, p, tail):
		# A variation of the conditional and the iteration
		initial_pc = p.pc
		subsearch = p.append(Instruction.SUBSEARCH)
		self.generate(next(s.arguments()), p, False)
		p.append(Instruction.NOFAIL)
		p.append(Instruction.JUMP, (initial_pc, ))
		subsearch.extra = p.pc

	def call(self, s, p, tail):
		# Strategy calls are executed by CALL instructions, but the body of
		# their definitions should be generated too. This is done afterwards
		# in the handle_calls method, which adapts the instruction arguments.
		p.append(Instruction.CALL, (s, tail))
		self.calls.append(p.pc - 1)

	def choice(self, s, p, tail):
		# Like a disjunction, but using the instruction CHOICE instead of JUMP
		choice_map, = s.arguments()
		branches, jumps = [], []
		choice_inst = p.append(Instruction.CHOICE)

		for branch in choice_map.arguments():
			w, b = branch.arguments()
			branches.append((self.m.downTerm(w), p.pc))
			self.generate(b, p, tail)
			jumps.append(p.append(Instruction.JUMP))

		jumps.pop()
		p.pop()

		choice_inst.extra = tuple(branches)
		for jump in jumps:
			jump.extra = (p.pc, )

	def sample(self, s, p, tail):
		variable, dist, strat = s.arguments()
		name, args = dist.arguments()

		# Parse distribution name and arguments
		name = str(name)[1:]
		args = tuple(map(self.m.downTerm, self.get_list(args, self.empty_term_list, self.term_list_symbol)))

		name = {
			'bernoulli': bernoulli_distrib,
			'uniform': random.uniform,
			'exp': random.expovariate,
			'norm': random.normalvariate,
			'gamma': random.gammavariate,
		}[name]

		p.append(Instruction.SAMPLE, (self.m.downTerm(variable), name, args))
		self.generate(strat, p, tail)
		p.append(Instruction.POP)

	def one(self, s, p, tail):
		# One can be ignored if desired
		if self.use_one:
			p.append(Instruction.ONE)
			self.generate(next(s.arguments()), p, False)
			p.append(Instruction.POP)

		else:
			self.generate(next(s.arguments()), p, tail)

	def stry(self, s, p, tail):
		cond, = s.arguments()

		self.make_conditional(cond, self.idle_const, self.idle_const, p, tail)

	def stest(self, s, p, tail):
		cond, = s.arguments()

		# Two nested subsearchs
		subsearch = p.append(Instruction.SUBSEARCH)
		self.make_conditional(cond, self.fail_const, self.idle_const, p, tail)
		p.append(Instruction.NOFAIL)
		p.append(Instruction.JUMP, ())

		subsearch.extra = p.pc

	def snot(self, s, p, tail):
		cond, = s.arguments()

		self.make_conditional(cond, self.fail_const, self.idle_const, p, tail)

	def compile(self, strat):
		"""Generate a program for the given strategy"""

		self.calls = []

		# Generate the code for the expression in a new program
		p = StratProgram()
		self.generate(strat, p, False)

		# A final POP instruction to communicate that a solution has been reached
		p.append(Instruction.POP)

		# Handle all strategy calls in strat and generate
		# (recursively) the code for their definitions
		self.handle_calls(p)

		if self.use_notify:
			return self.postprocess(p)

		return p

	def handle_calls(self, p):
		"""Handle strategy calls"""
		name_map, named_strats = {}, []

		for call_pc in self.calls:
			# Some optimizations would be possible (using unification in the most general case)
			callterm, tail = p[call_pc].extra
			name, args = callterm.arguments()

			# Get the arguments of the call
			args = list(map(self.m.downTerm, self.get_list(args, self.empty_term_list, self.term_list_symbol)))

			# Obtain the kinds of the call arguments to identify the strategy
			arg_kinds = [arg.getSort().kind() for arg in args]
			name = str(name)[1:]
			full_name = f'{name}({",".join(map(str, arg_kinds))})'

			# Check whether we have already compiled this strategy
			strat_nr, symbol = name_map.get(full_name, (None, None))

			if strat_nr is None:
				defs = []

				for sd in self.m.getStrategyDefinitions():
					domain_kinds = [tp.kind() for tp in sd.getStrategy().getDomain()]
					def_name = sd.getStrategy().getName()

					if def_name == name and arg_kinds == domain_kinds:
						lhs = sd.getLhs()
						defs.append((lhs, sd.getCondition(), p.pc))

						# Generate the body of the strategy definition
						self.generate(self.ml.upStrategy(sd.getRhs()), p, True)
						p.append(Instruction.POP)

						# The symbol is stored for building the call terms
						symbol = lhs.symbol()

				# This number will in call instructions to identify this strategy
				strat_nr = len(named_strats)

				name_map[full_name] = (strat_nr, symbol)
				named_strats.append((full_name, defs))

			# Strategies without definitions are fails
			if symbol is None:
				p.inst[call_pc] = Instruction(Instruction.JUMP, extra=())
			else:
				p[call_pc].extra = (symbol(*args), strat_nr, tail)

		p.defs = named_strats

	def generate(self, strat, p, tail):
		"""Append the code for the given strategy in the given program"""

		# The strategy combinator is found in the table
		symbol = strat.symbol()
		func = self.op_table.get(symbol)

		# If there is a handler for it, we call it
		if func is None:
			raise ValueError(f'unhandled strategy combinator {symbol}')

		getattr(self, func)(strat, p, tail)

	def postprocess(self, p):
		"""Postprocess the code to add NOTIFY and CHECKPOINT instructions"""

		# Addresses of all jump destinations in the code
		jump_dests = ({addr for inst in p.inst if inst.type == Instruction.JUMP for addr in inst.extra}
		           | {inst.extra for inst in p.inst if inst.type == Instruction.SUBSEARCH}
		           | {addr for inst in p.inst if inst.type == Instruction.CHOICE for _, addr in inst.extra})

		# For each definition, the address where it starts and the strategy it belongs to
		def_starts = sorted((addr, k) for k, (_, defs) in enumerate(p.defs) for _, _, addr in defs)
		def_starts, def_strats = zip(*def_starts) if def_starts else ([], [])

		class BasicBlock:
			"""Basic block of the program graph"""

			def __init__(self, start):
				self.start = start
				self.length = 1
				self.has_rewrite = False
				self.next = set()
				self.reachable = ('u',)

			def calls(self, p):
				"""Strategy that is called by this block (if any)"""

				last_inst = p[self.start + self.length - 1]
				return frozenset({last_inst.extra[1]}) if last_inst.type == Instruction.CALL else frozenset()

			def __repr__(self):
				return f'BasicBlock({self.start}, {self.length}, {self.has_rewrite}, {self.next}, {self.reachable})'

		def does_rewrite(inst):
			return inst.type == Instruction.RLAPP or inst.type == Instruction.RWCNEXT and inst.extra[0]

		# (1) Abstract the code as a graph of blocks linked by jumps

		# Blocks are aggregated by definition (the first entry is the initial expression)
		blocks = [[] for _ in range(len(def_starts) + 1)]
		# Table from line numbers to blocks
		block_map = {}

		# Iterate over the code to obtain the block structure
		current = BasicBlock(0)

		for d, df in enumerate(blocks):
			# Start and end of the expression code
			start = 0 if d == 0 else def_starts[d - 1]
			end = len(p.inst) if d == len(def_starts) else def_starts[d]

			for k in range(start, end):
				inst = p[k]

				# This block contains a rewrite
				if does_rewrite(inst):
					current.has_rewrite = True

				# JUMP, CHOICE, SUBSEARCH, and CALL instructions close a block
				if k + 1 in jump_dests or k + 1 == end or inst.type in (Instruction.JUMP, Instruction.CHOICE,
				                                                        Instruction.SUBSEARCH, Instruction.CALL):

					# The destinations of the JUMP are successors of this block
					if inst.type == Instruction.JUMP:
						for addr in inst.extra:
							current.next.add(addr)

					# The same happens with choice
					elif inst.type == Instruction.CHOICE:
						for _, addr in inst.extra:
							current.next.add(addr)

					# The negative branch is a successor
					elif inst.type == Instruction.SUBSEARCH:
						current.next.add(inst.extra)
						current.next.add(k + 1)

					# Otherwise, this is a fallthrough to the next block
					elif k + 1 != end:
						current.next.add(k + 1)

					# Starts a new block
					current.length = k - current.start + 1
					df.append(current)
					block_map[current.start] = current
					current = BasicBlock(k + 1)

		# (2) Calculate the reachability of each block from the beginning of its expression.
		# Unreachable blocks can be removed and knowing whether the end of the expression is
		# reachable without doing a rewrite is useful for deciding checkpoints.
		# (u=unreachable, r=reachable through a path that contains a rewrite,
		#  e=reachable through a path without rewrites)

		for fn in blocks:
			fn[0].reachable = ('r',) if fn[0].has_rewrite else ('e', {fn[0].calls(p)})

			# This is simply a depth-first search
			dfs_stack, visited = [fn[0]], {fn[0]}

			while dfs_stack:
				state = dfs_stack.pop()

				for next_state in state.next:
					next_state = block_map[next_state]

					# If next_state can be visited without passing through a rewrite
					if state.reachable[0] == 'e' and not next_state.has_rewrite:
						if next_state.reachable[0] != 'e':
							next_state.reachable = state.reachable
						else:
							# Strategies called within this block
							own_call = next_state.calls(p)
							# The contents of this set are sets of strategy numbers that
							# are called in path to next_state (only minimal sets are kept)
							new_set = ({s | own_call for s in next_state.reachable[1]
							            if not any(os < s for os in state.reachable[1])}
							         | {s | own_call for s in state.reachable[1]
							            if not any(os < s for os in next_state.reachable[1])})
							next_state.reachable = ('e', frozenset(new_set))

					if next_state not in visited:
						next_state.reachable = ('r',) if next_state.has_rewrite else state.reachable
						visited.add(next_state)
						dfs_stack.append(next_state)

		# The reachable field has been calculated assumming that calls do not apply rewrites,
		# but we can consider they do if every definition for it applies a rewrite

		# has_rewrite_strats holds for each strategy the definitions that do not apply a rewrite
		has_rewrite_strats = [set() for _ in range(len(p.defs))]

		for k, fn in enumerate(blocks[1:]):
			if fn[-1].reachable[0] == 'e':
				has_rewrite_strats[def_strats[k]].add(k)

		updated = {k for k, edefs in enumerate(has_rewrite_strats) if not edefs}

		while updated:
			new_updated = set()

			for k, fn in enumerate(blocks):
				if fn[-1].reachable[0] == 'e':
					fn[-1].reachable = ('e', {s for s in fn[-1].reachable[1] if s.isdisjoint(updated)})

					if not fn[-1].reachable[1]:
						fn[-1].reachable = ('r', )
						new_updated.add(k)

			updated = new_updated

		has_rewrite_strats = [not s for s in has_rewrite_strats]

		# Revisit the blocks and set has_rewrite taking calls into account
		for block in itertools.chain.from_iterable(blocks):
			called_strat = block.calls(p)

			if called_strat and has_rewrite_strats[next(iter(called_strat))]:
				block.has_rewrite = True

			# Remove the set of definitions, since it is no longer needed
			block.reachable = block.reachable[0]

		# Addresses where checkpoints should be inserted
		checkpoints = set()

		# (3) Find cycles without a rewrite in each definition and add checkpoints to prevent them

		for fn in blocks:
			cut_points, _ = find_cycles(fn[0], lambda b: (block_map[nx] for nx in b.next
			                                              if not block_map[nx].has_rewrite))
			checkpoints |= {cp.start for cp in cut_points}

		# Obtain the list of strategies called by each definition and whether the call is tail
		# (the key is the initial address of the definition)
		called = {fn[0].start: [p[k].extra[1:] for k in range(fn[0].start, fn[-1].start + fn[-1].length)
		                        if p[k].type == Instruction.CALL] for fn in blocks}

		# Find cycles without a rewrite between definitions and add checkpoints
		visited = set()

		for k, start in enumerate(def_starts):
			if start in visited:
				continue

			cut_points, visited_here = find_cycles(start, lambda n: (k for st, tail in called[n]
				if tail for _, _, k in p.defs[st][1])
				if blocks[def_starts.index(n) + 1][-1].reachable == 'e' else iter(()))

			checkpoints |= cut_points
			visited |= visited_here

		# (4) Decide where to add NOTIFY instructions

		notify_points = []  # Addresses where a notify should be added
		jump_over_notify = []  # Jumps that should be added to skip notifies
		# Blocks that carry a notify to the block in the key
		notifiers = {}
		# Blocks that jump to to the block in the key without address
		not_notifiers = {}

		# Backward jumps are always non-notifying
		for block in itertools.chain.from_iterable(blocks):
			last_address = block.start + block.length - 1
			inst = p[last_address]

			if inst.type == Instruction.JUMP:
				for addr in inst.extra:
					if addr < last_address:
						not_notifiers.setdefault(addr, set()).add(block)

			elif inst.type == Instruction.CHOICE:
				for _, addr in inst.extra:
					if addr < last_address:
						not_notifiers.setdefault(addr, set()).add(block)

		for fn in blocks:
			# Nesting level within rewriting conditions
			rwc_level = 0

			for bidx, block in enumerate(fn):
				# Inherited notify from incoming blocks
				notify_pending = block.start in notifiers and block.start not in not_notifiers

				# There is a notify pending, but it must be resolved right now
				if not notify_pending and block.start in notifiers:
					# The problem is that there are both notifiers and non-notifiers
					# jumping to this block. We must add a NOTIFY for the former that
					# the later must jump over. This is easy to achieve unless the
					# non-notifier is the previous block, which does not have a jump.

					# As an optimization, if there is a single notifier, notify at the
					# end of it (which may coincide with the beginning of this block,
					# but in this case the previous block is a notifier)
					our_notifiers = notifiers[block.start]
					notifier, *_ = our_notifiers
					if len(our_notifiers) == 1 and p[notifier.start + notifier.length - 1] in (Instruction.JUMP, Instruction.CHOICE):
						notify_points.append(notifier.start + notifier.length - 1)

					# Otherwise, notify here but not-notifiers should jump over it
					else:
						notify_points.append(block.start)

						# The previous block falls into this one, so it must skip
						# the NOTIFY with a new instruction
						if fn[bidx - 1] in not_notifiers[block.start] and \
						   p[block.start - 1].type not in (Instruction.JUMP, Instruction.CHOICE):
							jump_over_notify.append(block.start)

				# Review the instructions of the block
				for k in range(block.start, block.start + block.length):
					inst = p[k]

					# Take rewriting conditions into account for ignoring RLAPP there
					if inst.type == Instruction.RWCSTART:
						rwc_level += 1

					elif inst.type == Instruction.RWCNEXT and inst.extra[0]:
						rwc_level -= 1

					if notify_pending:
						# notify_pending is not possible within a rewriting condition

						# There is a RLAPP, so we need to issue a NOTIFY for the previous one
						if does_rewrite(inst):
							notify_points.append(k)

						# Failures discard the notification
						elif inst.type == Instruction.JUMP and not inst.extra:
							notify_pending = False

						# When any of the following instructions is found, we stop delaying the
						# notification since they involve branches or creating new contexts
						elif (inst.type == Instruction.JUMP and (len(inst.extra) > 1 or inst.extra[0] <= k) or
						      inst.type == Instruction.CHOICE and len(inst.extra) > 1 or
						      inst.type == Instruction.NEXTSUBTERM and not inst.extra[0] or
						      inst.type in (Instruction.CALL, Instruction.SUBSEARCH, Instruction.MATCHREW,
						                    Instruction.WMATCHREW, Instruction.RWCSTART, Instruction.SAMPLE)):
							notify_points.append(k)
							notify_pending = False

					elif rwc_level == 0 and does_rewrite(inst):
						notify_pending = True

				for sc in block.next:
					(notifiers if notify_pending else not_notifiers).setdefault(sc, set()).add(block)

			if notify_pending:
				notify_points.append(fn[-1].start + fn[-1].length - 1)

		# Rewrite the initial program adding checkpoints, removing unreachable blocks,
		# and adding NOTIFY instructions
		translation, new_pc = {}, 0
		need_regeneration = len(checkpoints) > 0 or len(notify_points) > 0

		checkpoints = ListWalker(sorted(checkpoints))
		notify_points = ListWalker(notify_points)
		jump_over_notify = ListWalker(jump_over_notify)

		# Calculate a translation of addresses (since new instructions are added)
		for block in itertools.chain.from_iterable(blocks):
			if block.reachable[0] == 'u':
				need_regeneration = True
				continue

			if jump_over_notify.next_equal(block.start):
				new_pc += 1

			translation[block.start] = new_pc
			new_pc += block.length

			if checkpoints.next_equal(block.start):
				new_pc += 1

			while notify_points.next_between(block.start, block.start + block.length):
				new_pc += 1

		# Return the program as is if no changes are needed
		if not need_regeneration:
			return p

		np = StratProgram()

		# Pending nested matchrew variables for the NOTIFY argument
		matchrew_vars = []

		# Reset the sequences to follow them again
		notify_points.reset()
		checkpoints.reset()
		jump_over_notify.reset()

		# Apply the already decided changes
		for block in itertools.chain.from_iterable(blocks):
			if block.reachable[0] == 'u':
				continue

			if checkpoints.next_equal(block.start):
				np.append(Instruction.CHECKPOINT)

			if jump_over_notify.next_equal(block.start):
				np.append(Instruction.JUMP, extra=(translation[block.start] + 1, ))

			def translate(addr):
				# Non-notifiers jumping to a NOTIFY must skip it
				if block in not_notifiers.get(addr, ()) and addr in notify_points:
					return translation[addr] + 1
				return translation[addr]

			# Translate addresses and add NOTIFY inside the block
			for k in range(block.start, block.start + block.length):
				inst = p[k]

				if notify_points.next_equal(k):
					np.append(Instruction.NOTIFY, extra=tuple(matchrew_vars))

				if inst.type in (Instruction.MATCHREW, Instruction.WMATCHREW):
					matchrew_vars.append(inst.extra[3])

				elif inst.type == Instruction.NEXTSUBTERM:
					if inst.extra[0]:
						matchrew_vars.pop()
					else:
						matchrew_vars[-1] = matchrew_vars[-1][1:]

				if inst.type == Instruction.JUMP:
					np.append(Instruction.JUMP, extra=tuple(translate(addr) for addr in inst.extra))

				elif inst.type == Instruction.SUBSEARCH:
					np.append(Instruction.SUBSEARCH, extra=translate(inst.extra))

				elif inst.type == Instruction.CHOICE:
					np.append(Instruction.CHOICE, extra=tuple((w, translate(pc)) for w, pc in inst.extra))

				else:
					np.inst.append(p[k])

		# Translate the addresses of the definitions
		np.defs = tuple((name, tuple((p, c, translation[addr]) for p, c, addr in defs)) for name, defs in p.defs)

		return np


class StackNode:
	"""Generic node in the execution stack (or tree)"""

	def __init__(self, parent=None, pc=None, seen=None, **kwargs):
		# Stack parent
		self.parent = parent
		# Set of seen states
		self.seen = parent.seen if seen is None else seen
		# Variable environment
		self.venv = kwargs['venv'] if 'venv' in kwargs else (None if parent is None else parent.venv)

		# Program counter to recover when this is popped (when returning from calls)
		self.pc = pc

	def already_seen(self, pc, term):
		"""Check whether a position and term have been previously visited"""

		if (pc, term, self.venv) in self.seen:
			return True

		self.seen.add((pc, term, self.venv))
		return False

	def already_seen_table(self, pc, term):
		"""Check whether a position and term have been previously visited"""

		# print('SEEN', (pc, term, self.venv), self.seen)

		return self.seen.get((pc, term, self.venv), None)

	def add_to_seen_table(self, pc, term, value):
		"""Add a position and term to the table of seen states"""

		self.seen[(pc, term, self.venv)] = value

	def __repr__(self):
		return f'StackNode({self.venv}, {self.pc})'


class RwcNode(StackNode):
	"""Stack node for rewriting conditions"""

	def __init__(self, parent=None, index=0, subs=None, context=None):
		super().__init__(parent=parent)

		# Index of the alternative conditional rule
		self.index = index
		# Substitution up to now
		self.subs = subs
		# Matching context
		self.context = context

	def __repr__(self):
		return f'RwcNode({self.venv}, index={self.index})'


class SubtermNode(StackNode):
	"""Stack node for subterm rewriting"""

	def __init__(self, parent=None, venv=None, context=None, pending=(), done=None, seen=None):
		super().__init__(parent=parent, venv=venv, seen=seen)

		# Matching context
		self.context = context
		# Accumulated substitution
		self.done = {} if done is None else done
		# Pending subterms
		self.pending = pending

	def progress(self, var, value, pc):
		"""Get the stack for the next subterm"""

		return type(self)(parent=self.parent, venv=self.venv, context=self.context,
		                  pending=self.pending[1:], done={**self.done, **{var: value}},
		                  seen=self.seen.fork((pc, value)))

	def __repr__(self):
		return f'SubtermNode({self.venv}, {self.context}, {self.done}, {self.pending})'


class OneStackNode(StackNode):
	"""Stack node for one"""

	def __init__(self, parent, pending_size):
		super().__init__(parent=parent)

		# Size of the pending stack
		self.pending_size = pending_size

	def __repr__(self):
		return f'OneStackNode({self.venv}, {self.pending_size})'


class HierarchicalSeenSet:
	"""Hierarchical set/table of seen states"""

	def __init__(self, seen):
		self.seen = seen
		self.children = {}

	def fork(self, pc):
		"""Create a new child subtable"""

		return self.children.setdefault(pc, HierarchicalSeenSet(type(self.seen)()))

	# Replicates the required methods of sets and dictionaries

	def __contains__(self, key):
		return key in self.seen

	def add(self, value):
		return self.seen.add(value)

	def __getitem__(self, key):
		return self.seen[key]

	def __setitem__(self, key, value):
		self.seen[key] = value

	def get(self, key, default=None):
		return self.seen.get(key, default)


class StratRunner:
	"""Runner for the strategy language machine"""

	class State:
		"""Execution state"""

		def __init__(self, term, pc, stack, conditional=False):
			# Program counter
			self.pc = pc
			# Current term
			self.term = term
			# Stack pointer
			self.stack = stack
			# Whether this state is for a conditional subsearch
			self.conditional = conditional

		def copy(self, term=None, pc=None, stack=None, conditional=False):
			"""Clone state with possibly some changes"""

			return StratRunner.State(
				self.term if term is None else term,
				self.pc + 1 if pc is None else pc,
				self.stack if stack is None else stack,
				conditional
			)

		def __repr__(self):
			return f'State({self.pc}, {self.term}, {self.stack})'

	def __init__(self, program, term, state_class=State, seen_class=set):
		# Program to be executed
		self.code = program
		# Root of the execution stack (tree)
		self.stack_root = StackNode(seen=HierarchicalSeenSet(seen_class()))
		# Current state being executed
		self.current_state = state_class(term, 0, self.stack_root)
		# Pending states
		self.pending = []

		# Class of states
		self.state_class = state_class
		# Class of the table of seen states
		self.seen_class = seen_class

		# Last solution to be found
		self.solution = None

		# Instruction handlers
		self.handlers = {
			Instruction.POP: self.pop,
			Instruction.JUMP: self.jump,
			Instruction.STRAT: self.strat,
			Instruction.RLAPP: self.rlapp,
			Instruction.TEST: self.test,
			Instruction.SUBSEARCH: self.subsearch,
			Instruction.NOFAIL: self.nofail,
			Instruction.CHOICE: self.choice,
			Instruction.CALL: self.call,
			Instruction.MATCHREW: self.matchrew,
			Instruction.NEXTSUBTERM: self.nextsubterm,
			Instruction.NOTIFY: self.notify,
			Instruction.RWCSTART: self.rwcstart,
			Instruction.RWCNEXT: self.rwcnext,
			Instruction.SAMPLE: self.sample,
			Instruction.WMATCHREW: self.wmatchrew,
			Instruction.ONE: self.one,
			Instruction.CHECKPOINT: self.checkpoint,
		}

	def reset(self, term):
		"""Reset the runner with a given initial term"""

		self.current_state = self.state_class(term, 0, self.stack_root)
		self.pending.clear()

	def dump_state(self):
		"""Dump current execution state (for debugging)"""

		print('-' * 80)
		print(f'\x1b[32m{self.current_state.pc}\x1b[0m', self.code[self.current_state.pc])
		print(self.current_state.term, '---', self.current_state.stack.venv)
		print(self.pending)

		stack = self.current_state.stack

		while stack is not None:
			print('', stack, id(stack.seen))
			stack = stack.parent

	def next_pending(self):
		"""Change the current state to the next pending state"""

		# If there is pending work, lets go for it
		while self.pending:
			self.current_state = self.pending.pop()

			# Conditional expressions push a conditional state for the negative
			# branch in the pending queue. It is only triggered if no execution
			# has succeeded for the condition, i.e. if the NOFAIL instruction
			# did not unset the pc attribute of the state.
			if self.current_state.conditional:
				if self.current_state.stack.pc is not None:
					# The state run outside the conditional stack node
					self.current_state.stack = self.current_state.stack.parent
					self.current_state.conditional = False
					return True
			else:
				return True

		# No more pending work for the strategy
		self.current_state = None
		return False

	def run(self):
		"""Run the strategy and get the next result"""

		self.solution = None

		# Keep running until the strategy is exhausted (or a solution is found)
		while self.current_state:
			# If the state is already visited, continue with other pending work
			while (self.current_state.stack.already_seen(self.current_state.pc,
			                                             self.current_state.term)):
				if not self.next_pending():
					return None

			# Current state
			state = self.current_state

			# The instruction to be executed
			inst = self.code[state.pc]

			self.handlers[inst.type](inst.extra, state.stack)

			if self.solution:
				return self.solution

		return None

	def pop(self, _, stack):
		"""Return from a strategy call or similar construct"""

		# Handle the one strategy
		if isinstance(stack, OneStackNode):
			# Discard all pending work (for the one argument)
			del self.pending[stack.pending_size:]
			self.current_state.pc += 1

		# Return from a strategy call
		elif stack.pc:
			self.current_state.pc = stack.pc

		# Pop the stack node
		if stack.parent:
			self.current_state.stack = self.current_state.stack.parent

		# This is the root node, so we have found a solution
		else:
			self.solution = self.current_state.term
			self.next_pending()

	def jump(self, args, stack):
		"""Jump to zero, one or more addresses"""

		# Zero options, this is a failure
		if not args:
			self.next_pending()
		else:
			# Change the program counter to the first option...
			self.current_state.pc = args[0]
			# ...and left the rest as pending states (in reverse order)
			self.pending += [self.current_state.copy(pc=npc) for npc in reversed(args[1:])]

	def strat(self, args, stack):
		"""Have a strategy executed by Maude"""

		# The order of the solutions is reversed (reversed may be used to avoid it)
		self.pending += [self.current_state.copy(term=sol)
		                 for sol, _ in self.current_state.term.srewrite(args)]

		# Go for the next pending state (which may have been generated by
		# the strategy in the previous line or not, if it has failed)
		self.next_pending()

	def get_rewrites(self, args, stack):
		"""Apply a rule and get the rewrites"""

		label, initial_subs, top = args

		# The initial substitution should be instantiated and reduced first
		if initial_subs:
			if stack.venv:
				initial_subs = {var: stack.venv.instantiate(value) for var, value in initial_subs.items()}

			for value in initial_subs.values():
				value.reduce()

			initial_subs = maude.Substitution(initial_subs)

		return self.current_state.term.apply(label,
		                                     initial_subs,
		                                     maxDepth=-1 if top else maude.UNBOUNDED)

	def rlapp(self, args, stack):
		"""Apply a rule"""

		self.pending += [self.current_state.copy(term=(t.reduce(), t)[1])
		                 for t, *_ in self.get_rewrites(args, stack)]

		self.next_pending()

	def test(self, args, stack):
		"""Test"""

		mtype, pattern, condition = args

		# Instantiate pattern and condition with the environment substitution
		if stack.venv is not None:
			pattern = stack.venv.instantiate(pattern)
			condition = instantiate_condition(condition, stack.venv)

		# A single match is enough to pass the test
		matched = next(self.current_state.term.match(pattern, condition=condition, maxDepth=mtype), None)

		if matched is None:
			self.next_pending()
		else:
			self.current_state.pc += 1

	def subsearch(self, args, stack):
		"""Start a subsearch by pushing a stack node with continuation"""

		# Push a new stack node with a fresh table of seen states to hold the subsearch
		# (the pc field will be cleared -in nofail- when a solution is found to disable the
		# execution of the negative branch).
		subsearch_stack = StackNode(parent=stack, pc=args, seen=HierarchicalSeenSet(self.seen_class()))
		self.current_state = self.current_state.copy(stack=subsearch_stack)

		# Push a new pending conditional state, which is only executed if all states after
		# it have failed, i.e. if the stack's pc has not been cleared.
		self.pending.append(self.current_state.copy(pc=args, stack=subsearch_stack, conditional=True))

	def nofail(self, args, stack):
		"""Pop the stack and discard the continuation on failure"""

		self.current_state.pc += 1
		# Abandon the subsearch
		self.current_state.stack = stack.parent
		# Disable the negative branch of the conditional
		stack.pc = None

	def choice(self, args, stack):
		"""Jump to an alternative according to the probabilities given by their weights"""

		weights, targets = zip(*args)
		weights = self.compute_weights(weights, stack.venv)

		try:
			self.current_state.pc, = random.choices(targets, weights)

		# All weights are zero
		except ValueError:
			self.next_pending()

	@staticmethod
	def compute_weights(weights, venv):
		"""Obtain the weights of a choice"""

		# Instantiate and reduce the weights
		if venv:
			weights = [venv.instantiate(w) for w in weights]

			for w in weights:
				w.reduce()

		# Convert the weights to Python's float
		# (unreduced weights will be silently converted to zero)
		weights = map(float, weights)

		return weights

	def do_call(self, args, stack):
		"""Generate the possible outcomes of a strategy call"""

		# Call term, index of the called strategy within the program, and whether it is tail
		call, strat_nr, tail = args
		# Definitions of the called named strategy
		_, defs = self.code.defs[strat_nr]

		# Instantiate the call term with the variable context
		if stack.venv:
			call = stack.venv.instantiate(call)
			call.reduce()

		# Match the call term with every definition
		pending = []

		for pattern, condition, addr in defs:
			# More than one match per call is possible
			for match, _ in call.match(pattern, condition):
				# In non-tail calls, the stack node is a child of the current node
				# and a fresh seen table is assigned because states should be
				# distinguished by its stack continuation.
				if not tail:
					new_stack = StackNode(parent=stack,
					                      pc=self.current_state.pc + 1,
					                      venv=match if len(match) > 0 else None,
					                      seen=stack.seen.fork(self.current_state.pc))
				# In tail calls, the stack node can be a sibling of the current one,
				# the return address is the same, and the seen table can be shared.
				else:
					new_stack = StackNode(parent=stack.parent,
					                      pc=stack.pc,
					                      venv=match if len(match) > 0 else None,
					                      seen=stack.seen)

				pending.append(self.current_state.copy(pc=addr, stack=new_stack))

		return pending

	def call(self, args, stack):
		"""Strategy call"""

		self.pending += self.do_call(args, stack)
		self.next_pending()

	def matchrew(self, args, stack):
		"""Matchrew"""

		mtype, pattern, condition, variables = args

		# Original pattern without instantiation
		original_pattern = pattern

		# Pattern and condition must be instantiated before matching
		if stack.venv:
			pattern = stack.venv.instantiate(pattern)
			condition = instantiate_condition(condition, stack.venv)

		for match, ctx in self.current_state.term.match(pattern, condition, maxDepth=mtype):
			merged_subs, context = self.process_match(stack.venv, match, ctx, original_pattern, variables)

			# Stack node that holds the information required for the matchrew
			new_stack = SubtermNode(parent=stack,
			                        venv=merged_subs,
			                        context=context,
			                        pending=[merged_subs.instantiate(var) for var in variables[1:]])

			# Start evaluating the first subterm
			self.pending.append(self.current_state.copy(term=merged_subs.instantiate(variables[0]),
			                                            stack=new_stack))

		self.next_pending()

	def wmatchrew(self, args, stack):
		"""Matchrew with weight"""

		mtype, pattern, condition, variables, weight = args

		# Original pattern without instantiation
		original_pattern = pattern

		# Pattern and condition must be instantiated before matching
		if stack.venv:
			pattern = stack.venv.instantiate(pattern)
			condition = instantiate_condition(condition, stack.venv)

		# Calculate the weight of every match and keep their substitutions and contexts
		match_data, weights = [], []

		for match, ctx in self.current_state.term.match(pattern, condition, maxDepth=mtype):
			merged_subs, context = self.process_match(stack.venv, match, ctx, original_pattern, variables)

			# Instantiated weight
			this_weight = merged_subs.instantiate(weight)
			this_weight.reduce()

			match_data.append((merged_subs, context))
			weights.append(float(this_weight))

		# Select one of the matches
		try:
			(merged_subs, context), = random.choices(match_data, weights)

		# All weights are zero or there are no weights
		except (ValueError, IndexError):
			self.next_pending()
			return

		# Stack node that holds the information required for the matchrew
		new_stack = SubtermNode(parent=stack,
		                        venv=merged_subs,
		                        context=context,
		                        pending=[merged_subs.instantiate(var) for var in variables[1:]])

		# Start evaluating the first subterm
		self.current_state = self.current_state.copy(term=merged_subs.instantiate(variables[0]),
			                                         stack=new_stack)

	@staticmethod
	def process_match(venv, match, ctx, pattern, variables):
		"""Instantiate the matching substitution in its context"""

		# Merged substitution
		merged_subs = merge_substitutions(match, venv)
		# Substitution without the variables of the subterms to be rewritten
		safe_subs = maude.Substitution({var: val for var, val in merged_subs
		                                if var not in variables}) if merged_subs else None

		# The term where the rewritten subterms will be finally filled
		context = ctx(safe_subs.instantiate(pattern) if safe_subs else pattern)

		return merged_subs, context

	def nextsubterm(self, args, stack):
		"""Finish a matchrew"""

		is_last, var = args

		if is_last:
			# Instantiate the pattern in context with the rewritten subterms
			self.current_state.term = maude.Substitution({**stack.done,
			                                              **{var: self.current_state.term}}).instantiate(stack.context)
			# Get out of the matchrew stack node
			self.current_state.stack = self.current_state.stack.parent
		else:
			# Progress to the next subterm and start rewriting it
			self.current_state.stack = self.current_state.stack.progress(var, self.current_state.term,
			                                                             self.current_state.pc)
			self.current_state.term = stack.pending[0]

		self.current_state.pc += 1

	def notify(self, args, stack):
		"""Notify a rewrite"""

		# Do nothing (subclasses will)
		self.current_state.pc += 1

	def rwcstart(self, args, stack):
		"""Start a rewriting condition search"""

		top, initial_subs, candidates = args
		stack = self.current_state.stack
		module = self.current_state.term.symbol().getModule()
		initial_subs_obj = None

		# The initial substitution should be instantiated and reduced first
		if initial_subs:
			if stack.venv:
				initial_subs = {var: stack.venv.instantiate(value) for var, value in initial_subs.items()}

			for value in initial_subs.values():
				value.reduce()

			initial_subs_obj = maude.Substitution(initial_subs)

		for k, (lhs, initial_cond, rwc_lhs) in enumerate(candidates):
			hole = module.parseTerm(f'%hole:{lhs.getSort()}')

			# The initial substitution affects the condition and left-hand side of the rule
			if initial_subs:
				lhs = initial_subs_obj.instantiate(lhs)
				initial_cond = instantiate_condition(initial_cond, initial_subs_obj)

			for match, ctx in self.current_state.term.match(lhs, initial_cond, maxDepth=-1 if top else maude.UNBOUNDED):
				# Substitution for the rule application (containing the initial and matching variables)
				subs = maude.Substitution({**dict(match), **initial_subs})
				# Stack node holding the context and accumulated substitution of the rewriting condition
				new_stack = RwcNode(parent=stack, index=k, subs=subs, context=(ctx(hole), hole))
				# Start rewriting the left-hand side of the first condition fragment
				new_term = subs.instantiate(rwc_lhs)
				new_term.reduce()
				self.pending.append(self.current_state.copy(term=new_term, stack=new_stack))

		self.next_pending()

	def rwcnext(self, args, stack):
		"""Continue a rewriting condition search"""

		is_final, candidates = args
		stack = self.current_state.stack

		# rwc_lhs is the LHS of the next condition fragment
		# or the RHS of the rule if is_final
		rwc_rhs, condition, rwc_lhs = candidates[stack.index]
		rwc_rhs = stack.subs.instantiate(rwc_rhs)

		for match, _ in self.current_state.term.match(rwc_rhs, instantiate_condition(condition, stack.subs)):
			# Accumulate the new matched variables in the substitution
			subs = merge_substitutions(match, stack.subs)

			if is_final:
				context, hole = stack.context
				new_term = maude.Substitution({hole: subs.instantiate(rwc_lhs)}).instantiate(context)
				new_term.reduce()
				self.pending.append(self.current_state.copy(term=new_term, stack=stack.parent))
			else:
				new_stack = RwcNode(parent=stack.parent, index=stack.index, subs=subs, context=stack.context)
				new_term = subs.instantiate(rwc_lhs)
				new_term.reduce()
				self.pending.append(self.current_state.copy(term=new_term, stack=new_stack))

		self.next_pending()

	def sample(self, args, stack):
		"""Sample a probability distribution into a variable"""

		variable, dist, dargs = args

		# Arguments of the distribution instantiated in the variable context
		dargs = [(stack.venv.instantiate(arg) if stack.venv else arg) for arg in dargs]

		# Reduce them
		for arg in dargs:
			arg.reduce()

		dargs = [float(arg) for arg in dargs]

		new_variable = {variable: variable.symbol().getModule().parseTerm(str(dist(*dargs)))}

		self.current_state.pc += 1
		self.current_state.stack = StackNode(parent=self.current_state.stack,
		                                     venv=merge_substitutions(stack.venv, new_variable))

	def one(self, args, stack):
		"""Enter a subsearch for a single solution of the substrategy"""

		self.current_state.pc += 1
		# Create a stack node to save the current number of pending states,
		# so that we can discard all new states when a solution is found
		self.current_state.stack = OneStackNode(stack, len(self.pending) - 1)

	def checkpoint(self, args, stack):
		"""Check a loop without rewrites"""

		# Do nothing (subclasses will)
		self.current_state.pc += 1


def rebuild_term(term, stack, args):
	"""Rebuild the term within nested matchrews"""

	# If we are rewriting a subterm, we must rebuild the complete
	# term going up subterm stack nodes.
	new_term, mrew_stack = term, stack

	# The argument of NOTIFY is a list of variables of the pending
	# subterms for every enclosing matchrew operator
	for pending_vars in args:
		# Find the next subterm stack node
		while not isinstance(mrew_stack, SubtermNode):
			mrew_stack = mrew_stack.parent

		# If the context is a single variable, there is nothing to do
		if mrew_stack.context.isVariable():
			continue

		# Substitution with...
		subs = maude.Substitution({
			# ...the subterms that have already been rewritten, and...
			**mrew_stack.done,
			# ..the current and the pending subterms.
			**dict(zip(pending_vars, [new_term] + mrew_stack.pending))
		})
		# Rebuild the subterm
		new_term = subs.instantiate(mrew_stack.context)

	# The new term should be reduced
	new_term.reduce()

	return new_term


class GraphExecutionState(StratRunner.State):
	"""Execution state for generating a graph"""

	def __init__(self, term, pc, stack, conditional=False, graph_node=None, extra=None):
		super().__init__(term, pc, stack, conditional)
		# Node in the rewrite graph that is being built
		self.graph_node = graph_node
		# Additional information for the last rewrite
		self.extra = extra

	def copy(self, term=None, pc=None, stack=None, conditional=False, graph_node=None, extra=None):
		"""Clone state with possibly some changes"""

		return GraphExecutionState(
			self.term if term is None else term,
			self.pc + 1 if pc is None else pc,
			self.stack if stack is None else stack,
			conditional,
			self.graph_node if graph_node is None else graph_node,
			self.extra if extra is None else extra,
		)


class BadProbStrategy(Exception):
	"""Bad probabilitistic strategy that cannot engender MDPs or DTMCs"""

	def __init__(self):
		super().__init__('Strategies are not allowed to make nondeterministic choices '
		                 'between quantified ones and a rewrite. No DTMC or MDP can be derived.')


class SubsearchNode(StackNode):
	"""Stack node for subsearches"""

	def __init__(self, dfs_base=None, subsearch_id=None, **kwargs):
		super().__init__(**kwargs)

		# Base of the DFS stack when the search started
		self.dfs_base = dfs_base
		# Identifier of the subsearch
		self.subsearch_id = subsearch_id

	def __repr__(self):
		return f'SubsearchNode(id={self.subsearch_id}, dfs_base={self.dfs_base})'


class MarkovRunner(StratRunner):
	"""Runner that extracts a DTMC or MDP"""

	class GraphState:
		"""State of the graph

		In addition to the actual states of the graph, there may be temporary fake
		states used to gather the evolutions of the branches of a choice. They are
		characterized by a None value in their term attribute.
		"""

		def __init__(self, term):
			self.term = term
			# Set of graph states that follow by a rewrite
			self.children = set()
			# Set of choice alternatives (each an association of
			# weights or probabilities to graph states)
			self.child_choices = set()
			# Temporary information about the last rewrite
			self.last_rewrite = None
			self.actions = {}
			self.solution = False
			self.valid = True
			# Each graph state has its own list of pending execution states
			self.pending = []
			# Subsearches for which a solution is reachable from here
			self.solve_subsearch = set()

		def __repr__(self):
			return f'GraphState({self.term}, {self.children}, {self.child_choices}, {self.solution})'

	def __init__(self, program, term):
		super().__init__(program, term, state_class=GraphExecutionState, seen_class=dict)
		self.root_node = self.GraphState(term)
		self.dfs_stack = [self.root_node]

		# A dictionary from execution states to graph states: the latter
		# should be pushed to the DFS stack when the former are executed
		self.push_state = {}

		# Solution states, indexed by term (we introduce them as children
		# of graph states where both a solution of the strategy and a
		# different sucessor can be reached)
		self.solution_states = {}

	def resolve_choice(self, choice, actions):
		"""Resolve a choice with weights into probabilities"""

		# While choice is a sequence of (weight, state), new_choice is
		# a dictionary from state to probability
		new_choice = {}

		for w, s in choice:
			if not s.valid:
				continue

			# If s is not a fake state
			if s.term is not None:
				old_w = new_choice.get(s, 0.0)
				new_choice[s] = old_w + w

			else:
				# Nondeteterministic and choice alternatives
				nd_opts, ch_opts = len(s.children), len(s.child_choices)

				if nd_opts + ch_opts > 1:
					raise BadProbStrategy

				# Copy the action information
				for target, info in s.actions.items():
					old_info = actions.get(target)

					if old_info:
						old_info.extend(info)
					else:
						actions[target] = info

				if nd_opts:
					s, = s.children
					old_w = new_choice.get(s, 0.0)
					new_choice[s] = old_w + w

				if ch_opts:
					child_choice, = s.child_choices
					total_cw = sum(child_choice.values())
					for cs, cw in child_choice.items():
						old_w = new_choice.get(cs, 0.0)
						new_choice[cs] = old_w + cw * w / total_cw

		return new_choice

	def get_solution_state(self, term):
		"""Get the solution state for the given term"""

		solution_state = self.solution_states.get(term)

		if not solution_state:
			solution_state = self.GraphState(term)
			self.solution_states[term] = solution_state

		return solution_state

	def next_pending(self):
		"""Change the current state to the next pending state"""

		# While the DFS is still alive
		while self.dfs_stack:
			graph_state = self.dfs_stack[-1]

			# If there is pending work, lets go for it
			while graph_state.pending:
				self.current_state = self.pending.pop()
				push_state = self.push_state.get(self.current_state)

				# The same as StratRunner.next_pending
				if self.current_state.conditional:
					if self.current_state.stack.pc is not None:
						self.current_state.stack = self.current_state.stack.parent
						self.current_state.conditional = False
						return True

				# This execution state is linked to a fake choice-generated graph state
				# that should be pused to the DFS stack
				elif push_state:
					self.dfs_stack.append(push_state)
					self.pending = push_state.pending
					return True

				else:
					return True

			# Check whether the graph state is valid
			graph_state.valid = graph_state.solution or graph_state.children or graph_state.child_choices

			# Link a solution state if there are children
			# (these states are needed to explicit the self-loop for the stuttering
			# extension of finite traces without introducing spurious executions)
			if graph_state.solution and (graph_state.children or graph_state.child_choices) and \
			   graph_state not in graph_state.children:
				solution_state = self.get_solution_state(graph_state.term)
				graph_state.children.add(solution_state)
				graph_state.actions[solution_state] = None

			# Adjust the probilities of the choice operators
			new_choices = []

			for choice in graph_state.child_choices:
				resolved_choice = self.resolve_choice(choice, graph_state.actions)

				if len(resolved_choice) == 1:
					child, = resolved_choice.keys()
					graph_state.children.add(child)

				elif resolved_choice:
					new_choices.append(resolved_choice)

			graph_state.child_choices = tuple(new_choices)

			self.dfs_stack.pop()

			if self.dfs_stack:
				dfs_top = self.dfs_stack[-1]
				self.pending = dfs_top.pending

				# Add graph_state as a child if it is valid
				if graph_state.valid and graph_state.term is not None:
					dfs_top.children.add(graph_state)
					dfs_top.actions.setdefault(graph_state, []).append(dfs_top.last_rewrite)
					dfs_top.last_rewrite = None  # unneeded, but cleaner

		# No more pending work for the strategy
		self.current_state = None
		return False

	def pop(self, _, stack):
		"""Return from a strategy call or similar construct"""

		# Return from a strategy call
		if stack.pc:
			self.current_state.pc = stack.pc

		# Actually pop the stack node
		if stack.parent:
			self.current_state.stack = self.current_state.stack.parent

		# This is the root node, so we have found a solution and mark the state
		else:
			graph_state = self.dfs_stack[-1]

			# Fake states are not marked as solutions but added a
			# solution state as a child (this is also done later to
			# marked states unless they do not have successors)
			if graph_state.term is None:
				graph_state.children.add(self.get_solution_state(self.current_state.term))
			else:
				graph_state.solution = True

			self.next_pending()

	def choice(self, args, stack):
		"""A choice node"""

		weights, targets = zip(*args)
		weights = list(self.compute_weights(weights, stack.venv))

		# Remove options with null weight
		targets = [target for k, target in enumerate(targets) if weights[k] != 0.0]
		weights = [w for w in weights if w != 0.0]

		# If there is only a positive weight, we can proceed with it
		if len(weights) == 1:
			self.current_state.pc = targets[0]
			return

		# Otherwise, if there is at least one
		if weights:
			graph_state = self.dfs_stack[-1]

			# Create a new graph state (without term) for each branch of the choice
			new_states = [self.GraphState(None) for _ in range(len(weights))]

			# These graph states cannot be pushed to the DFS stack (because we
			# are not exploring them yet), they should be pushed when the
			# execution state new_xs is executed, so we add them to push_state
			for k, target in enumerate(targets):
				new_xs = self.current_state.copy(pc=target)
				self.pending.append(new_xs)
				self.push_state[new_xs] = new_states[k]

			# Add the resolved choice to the graph state
			graph_state.child_choices.add(tuple(zip(weights, new_states)))

		self.next_pending()

	def wmatchrew(self, args, stack):
		"""A matchrew with weight node"""

		mtype, pattern, condition, variables, weight = args

		# Original pattern without instantiation
		original_pattern = pattern

		# Pattern and condition must be instantiated before matching
		if stack.venv:
			pattern = stack.venv.instantiate(pattern)
			condition = instantiate_condition(condition, stack.venv)

		# Calculate the weight of every match and keep their substitutions and contexts
		targets, weights = [], []

		for match, ctx in self.current_state.term.match(pattern, condition, maxDepth=mtype):
			merged_subs, context = self.process_match(stack.venv, match, ctx, original_pattern, variables)

			# Instantiated weight
			this_weight = merged_subs.instantiate(weight)
			this_weight.reduce()
			this_weight = float(this_weight)

			if this_weight > 0.0:
				targets.append((merged_subs, context))
				weights.append(float(this_weight))

		# Execution states that start evaluating the first subterm
		# with stacks that keep the information for the matchrew
		new_xss = [self.current_state.copy(term=merged_subs.instantiate(variables[0]),
		                                   stack=SubtermNode(parent=stack,
		                                                     venv=merged_subs,
		                                                     context=context,
		                                                     pending=[merged_subs.instantiate(var)
		                                                              for var in variables[1:]]))
		           for merged_subs, context in targets]

		# If there is only a positive weight, we can proceed with it
		if len(weights) == 1:
			self.current_state = new_xss[0]
			return

		# Otherwise, if there is at least one
		if weights:
			graph_state = self.dfs_stack[-1]

			# Create a new graph state (without term) for each branch of the matchrew
			new_states = [self.GraphState(None) for _ in range(len(weights))]

			# These graph states cannot be pushed to the DFS stack (because we
			# are not exploring them yet), they should be pushed when the
			# execution state new_xss[k] is executed, so we add them to push_state
			for k in range(len(targets)):
				self.pending.append(new_xss[k])
				self.push_state[new_xss[k]] = new_states[k]

			# Add the resolved choice to the graph state
			graph_state.child_choices.add(tuple(zip(weights, new_states)))

		self.next_pending()

	def notify(self, args, stack):
		"""Record a transition in the graph"""

		state = self.current_state
		state.pc += 1

		# Check whether this state has already been visited
		successor = stack.already_seen_table(state.pc, state.term)

		# This is a new state, so add it to the graph
		if successor is None:
			new_term = rebuild_term(state.term, stack, args)

			new_state = self.GraphState(new_term)
			stack.add_to_seen_table(state.pc, state.term, new_state)
			self.dfs_stack.append(new_state)
			self.pending = new_state.pending

			# The extra information for the last rewrite is recorded
			self.dfs_stack[-2].last_rewrite = state.extra

		else:
			if successor.valid:
				dfs_top = self.dfs_stack[-1]
				dfs_top.children.add(successor)
				dfs_top.actions.setdefault(successor, []).append(state.extra)

			# Handle the deactivation of the negative branch of
			# a subsearch when this state leaded to a solution
			to_be_solved = len(successor.solve_subsearch)

			while stack and to_be_solved:
				subsearch_id = getattr(stack, 'subsearch_id', None)
				if subsearch_id in successor.solve_subsearch:
					stack.pc = None
					to_be_solved -= 1
				stack = stack.parent

			self.next_pending()

	def checkpoint(self, args, stack):
		"""Check whether this state has been visited and interrupt the execution"""

		if stack.already_seen_table(self.current_state.pc, self.dfs_stack[-1]):
			self.next_pending()

		else:
			stack.add_to_seen_table(self.current_state.pc, self.dfs_stack[-1], True)
			self.current_state.pc += 1

	def subsearch(self, args, stack):
		"""Start a subsearch by pushing a stack node with continuation"""

		# The same as StratRunner.subsearch, but forking the seen set. Moreover,
		# the size of the DFS stack is recorded to allow finding out which states
		# lead to solutions within the search (identified by the current PC)
		subsearch_stack = SubsearchNode(parent=stack, pc=args,
		                                seen=stack.seen.fork(self.current_state.pc),
		                                dfs_base=len(self.dfs_stack),
		                                subsearch_id=self.current_state.pc)
		self.current_state = self.current_state.copy(stack=subsearch_stack)

		# Exactly the same as StratRunner.subsearch
		self.pending.append(self.current_state.copy(pc=args, stack=subsearch_stack, conditional=True))

	def nofail(self, args, stack):
		"""Pop the stack and discard the continuation on failure"""

		# Annotate the graph states with the current subsearch,
		# for which they provides a solution
		for node in self.dfs_stack[stack.dfs_base:]:
			node.solve_subsearch.add(stack.subsearch_id)

		super().nofail(args, stack)

	def rlapp(self, args, stack):
		"""Apply a rule"""

		self.pending += [self.current_state.copy(term=(t.reduce(), t)[1], extra=rl)
		                 for t, _, _, rl in self.get_rewrites(args, stack)]

		self.next_pending()

	def run(self):
		"""Run the strategy to generate the graph"""

		self.solution = None
		self.pending = self.root_node.pending

		while self.current_state:
			state = self.current_state

			# The instruction to be executed
			inst = self.code[state.pc]

			self.handlers[inst.type](inst.extra, state.stack)

		return self.root_node


class MetadataRunner(MarkovRunner):
	"""Runner for the metadata assignment method with non-ground weights"""

	# Choice and matchrew with weight are simulated to match
	# the behavior when all metadata weights are ground
	choice = StratRunner.choice
	wmatchrew = StratRunner.wmatchrew

	def __init__(self, program, term, stmt_map):
		super().__init__(program, term)
		self.stmt_map = stmt_map

	def rlapp(self, args, stack):
		"""Apply a rule"""

		for t, sb, _, rl in self.get_rewrites(args, stack):
			# Term.apply does not normalize output terms
			t.reduce()

			# Evaluate the metadata weight
			weight = self.stmt_map.get(rl, 1.0)

			# If it is not a literal, we should reduce the term
			if not isinstance(weight, float):
				weight = sb.instantiate(weight)
				weight.reduce()
				weight = float(weight)

			self.pending.append(self.current_state.copy(term=t, extra=(rl, weight)))

		self.next_pending()


class RandomRunner(StratRunner):
	"""Runner that resolves every choice locally at random without backtracking.

	Instead of solutions of the strategy, the run method returns the succesive steps
	of a single random rewriting path. Hence, conditionals do no work with this runner,
	since exploration is not exhaustive, and failed executions are not discarded."""

	def __init__(self, program, term):
		super().__init__(program, term)

	def detect_nondeterminism(self, usermsgs):
		"""Discover local nondeterminism, rewrites the program, and warns about it"""

		# Lines where the body of a strategy definition starts
		entry_points = sorted(((line, name) for name, defs in self.code.defs for _, _, line in defs), reverse=True)
		current_name = 'the given expression'
		# Detect trivial subsearches
		subsearches, trivial_subsearch = [], False

		for k, inst in enumerate(self.code.inst):
			# Jumps with multiple desinations
			if inst.type == Instruction.JUMP and len(inst.extra) > 1:
				inst.extra = inst.extra[:1]
				usermsgs.print_warning(f'Unquantified nondeterminism detected in {current_name}. '
				                       'It will be resolved arbitrarily.')

			# Conditional expressions
			elif inst.type == Instruction.SUBSEARCH:
				subsearches.append(k)
				trivial_subsearch = True

			elif inst.type == Instruction.NOFAIL:
				last_subsearch = subsearches.pop()

				if not trivial_subsearch:
					self.code.inst[last_subsearch].type = Instruction.JUMP
					self.code.inst[last_subsearch].extra = (last_subsearch + 1,)
					usermsgs.print_warning(f'Conditional expression detected in {current_name}. '
					                       'Its semantics may not be respected.')

				trivial_subsearch = False

			# Update the current procedure name
			elif inst.type == Instruction.POP:
				if entry_points and entry_points[-1][0] == k + 1:
					current_name = f'strategy {entry_points[-1][1]}'
					entry_points.pop()

			# Trivial subsearches can only contain tests
			if trivial_subsearch and inst.type not in (Instruction.TEST, Instruction.SUBSEARCH):
				trivial_subsearch = False

	def next_pending(self):
		"""Change the current state a random pending state"""

		# If there is no pending work, the strategy is exhausted
		if not self.pending:
			self.current_state = None
			return False

		# Otherwise, a pending state is chosen at random and all other pending
		# states are discarded. Indeed, pending states are not actually pending but
		# they have just been generated by a nondeterministic instruction (rule
		# application, call, matchrew...) before calling this method. Reimplemeting
		# these instructions is avoided by this trick.
		self.current_state = random.choice(self.pending)
		self.pending.clear()

		return True

	def jump(self, args, stack):
		"""Takes a branch of the jump at random"""

		if not args:
			self.current_state = None
		else:
			self.current_state.pc = random.choice(args)

	def notify(self, args, stack):
		"""Notify the rewrite as a solution"""

		# The notion of "solution" in this execution mode is "step" or "rewrite"
		self.solution = rebuild_term(self.current_state.term, stack, args)
		self.current_state.pc += 1

	def subsearch(self, args, stack):
		"""Handle trivial subsearches"""

		# For subsearch and nofail, we are assuming that
		# detect_nondeterminism has been called (*)

		# The next instruction must be a test by (*)
		state = self.current_state
		state.pc += 1
		inst = self.code.inst[state.pc]

		while inst.type == Instruction.TEST:
			self.test(inst.extra, stack)

			# If the test succeeds the execution continues,
			# otherwise the negative branch is run
			if self.current_state:
				inst = self.code.inst[state.pc]

			else:
				state.pc = args
				self.current_state = state
				return

		# This should not happen by (*)
		if inst.type != Instruction.NOFAIL:
			self.current_state = None

	def nofail(self, args, stack):
		"""Subsearches are not opened, so it is enough to increment the PC"""

		self.current_state.pc += 1

	def pop(self, _, stack):
		"""Return from a strategy call or similar construct"""

		# Return from a strategy call
		if stack.pc:
			self.current_state.pc = stack.pc

		# Pop the stack node
		if stack.parent:
			self.current_state.stack = self.current_state.stack.parent

		# This is the root node, we have finished
		else:
			self.current_state = None

	def run(self):
		"""Run the strategy and get the next step"""

		self.solution = None

		while self.current_state:
			# Visited states are not tracked, we are simulating

			# The instruction to be executed
			inst = self.code[self.current_state.pc]
			self.handlers[inst.type](inst.extra, self.current_state.stack)

			if self.solution:
				return self.solution

		return None
