#
# Common functions to parse and manipulate formulae in Python-list form
#

from .common import usermsgs, maude
from .resources import get_templog


def collect_aprops(form, aprops):
	"""Collect the atomic propositions in a formula"""
	head, *rest = form

	if head == 'Prop':
		aprops.add(rest[0])

	elif head in {'<_>_', '`[_`]_'}:
		collect_aprops(rest[1], aprops)

	elif head != 'Var':
		for arg in rest:
			collect_aprops(arg, aprops)


def _actionParse(aprop):
	"""Convert an action to string removing the type annotation"""
	text = str(aprop)
	type_annot = text.find(').@ActionList@')

	return text[1:type_annot] if text[0] == '(' and type_annot > 0 else text


def _actionSpecParse(acspec, labels):
	"""Parse an element of the @ActionSpec@ sort as a Python list of strings"""
	if str(acspec.symbol()) == '~_':
		actions = _actionSpecParse(next(acspec.arguments()), [])
		return list(labels - set(actions))

	elif str(acspec.symbol()) == '__':
		return [_actionParse(action) for action in acspec.arguments()]

	else:
		return [_actionParse(acspec)]


def _formula2List(form, *extra_args):
	"""Convert the formula parsed by Maude into an pure Python list"""
	prop_sort, variable_sort, labels = extra_args

	if form.getSort() <= prop_sort:
		return ['Prop', str(form)]

	elif form.getSort() <= variable_sort:
		return ['Var', str(form)]

	elif str(form.symbol()) in ['<_>_', '`[_`]_']:
		symbol_args = list(form.arguments())
		args = [_actionSpecParse(symbol_args[0], labels), _formula2List(symbol_args[1], *extra_args)]

	else:
		args = [_formula2List(x, *extra_args) for x in form.arguments()]

	return [str(form.symbol())] + args


class Parser:
	"""
	Parser of any temporal formula supported by umaudemc

	Since the parsing module depends on the formula itself, there is no
	much we can catch between executions.
	"""

	def __init__(self):
		# The module with the parsing module generator
		self.templog = None
		self.makeParserModule = None
		self.get_templog()

		self.labels = None
		self.rule_labels = None
		self.base_module = None

	def is_ok(self):
		"""Whether the parser has been loaded succesfully"""
		return self.makeParserModule is not None

	def set_module(self, module, metamodule=None):
		"""Set the module where formulae should be parsed"""

		# Build a list with the rule labels of the module (to allow
		# using them as edge labels in mu-calculus formulae)
		# (opaque strategies are not being considered)
		self.labels = [rl.getLabel() for rl in module.getRules()
		               if rl.getLabel() is not None]
		self.rule_labels = ' '.join([f"'{label}" for label in self.labels]) \
				   if self.labels != [] else '(nil).QidList'

		# Store the module as a metaterm in templog
		self.base_module = self.templog.parseTerm(f"upModule('{str(module)}, false)"
		                                          if metamodule is None else str(metamodule))

	def get_templog(self):
		"""Set up the parse module generator"""

		self.templog = maude.getModule('TEMPORAL-LOGIC-META')

		# We load it if not already loaded
		if not self.templog:
			# The model-checker.maude file is only loaded once to prevent advisory messages
			# about modules being redefined (sload is not enough if a model-checker.maude file
			# is present in the directory of the input file)
			if not maude.getModule('STRATEGY-MODEL-CHECKER'):
				maude.load('model-checker')

			# Loads the temporal logic parser
			with get_templog() as templog_path:
				if not maude.load(str(templog_path)):
					usermsgs.print_error('Error loading temporal logic parser (templog.maude).')
					return False

			self.templog = maude.getModule('TEMPORAL-LOGIC-META')

			if not self.templog:
				usermsgs.print_error('Error loading temporal logic parser (TEMPORAL-LOGIC-META module).')
				return False

		module_kind = self.templog.findSort('Module').kind()
		qidlist_kind = self.templog.findSort('QidList').kind()

		if not module_kind or not qidlist_kind:
			usermsgs.print_error('Error loading temporal logic parser (bad TEMPORAL-LOGIC-META module).')
			return False

		self.makeParserModule = self.templog.findSymbol('makeParserModule',
		                                                [module_kind, qidlist_kind, qidlist_kind],
		                                                module_kind)

		if not self.makeParserModule:
			usermsgs.print_error('Error loading temporal logic parser (missing makeParserModule operator).')

	def deduce_logic(self, extmod, formula):
		"""Deduce the least-general logic a formula belongs to"""

		formula_kind = extmod.findSort('Formula').kind()
		formelems_kind = extmod.findSort('@TmpFormElems@').kind()
		logicname_kind = extmod.findSort('@TmpLogicName@').kind()

		if formula_kind is None or formelems_kind is None or logicname_kind is None:
			usermsgs.print_error('Missing expected types in the formula parsing module.'
			                     'Check the templog.maude file.')
			return None

		# Get the operators to deducing its type and collect its atomic props
		formulaElems = extmod.findSymbol('formulaElems', [formula_kind], formelems_kind)
		formulaType = extmod.findSymbol('formulaType', [formelems_kind], logicname_kind)

		if formulaElems is None or formulaType is None:
			usermsgs.print_error('Missing expected operators in the formula parsing module.'
			                     'Check the templog.maude file.')
			return None

		formulaLogic = formulaType(formulaElems(formula))
		formulaLogic.reduce()

		return str(formulaLogic)

	def parse(self, formula):
		"""Parse a temporal formula from LTL to mu-calculus"""

		# Make a module where to parse the temporal formula
		parser_metamod = self.makeParserModule(
			self.base_module,
			self.templog.parseTerm('tokenize("{}")'.format(formula.replace('"', '\"'))),
			self.templog.parseTerm(self.rule_labels)
		)
		parser_metamod.reduce()
		extmod = maude.downModule(parser_metamod)

		formula_kind = extmod.findSort('Formula').kind()

		# Parses the temporal formula and deduces its type
		parsed_formula = extmod.parseTerm(formula, formula_kind)
		# a copy can be made to preserve the original

		if parsed_formula is None:
			usermsgs.print_error('The given temporal formula cannot be parsed.')
			return None, None

		logic = self.deduce_logic(extmod, parsed_formula)

		if logic is None:
			return None, None

		mcvariable_sort = extmod.findSort('@MCVariable@')
		prop_sort = extmod.findSort('Prop')

		if mcvariable_sort is None or prop_sort is None:
			usermsgs.print_error('Missing expected types in the formula parsing module.'
			                     'Check the templog.maude file.')
			return None, None

		return _formula2List(parsed_formula, prop_sort, mcvariable_sort, set(self.labels)), logic
