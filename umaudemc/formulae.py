#
# Common functions to parse and manipulate formulae in Python-list form
#

from .common import usermsgs, maude
from .resources import get_resource_path


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


def _add_path_premise_ctl(form, premise):
	"""Add a premise to every path quantification in CTL* formulae"""
	head, *rest = form

	if head == 'A_' or head == 'E_':
		bop = '_->_' if head == 'A_' else '_/\\_'
		return [head, [bop, premise, _add_path_premise_ctl(rest[0], premise)]]

	elif head == 'Prop':
		return form

	else:
		return [head] + [_add_path_premise_ctl(subf, premise) for subf in rest]


def add_path_premise(form, premise, logic):
	"""Add a premise that limits the paths under consideration"""

	if logic == 'LTL':
		return ['_->_', premise, form]

	elif logic.startswith('CTL'):
		return _add_path_premise_ctl(form, premise)

	else:
		return form


def formula_list2str(formula):
	"""Convert a formula given as a Python list to a string"""

	head, *rest = formula

	if head == 'Prop':
		return rest[0]
	elif head == 'Var':
		return rest[0]
	else:
		rest_str = [f' ({formula_list2str(arg)}) ' for arg in rest]
		rest_it = iter(rest_str)
		g = ''

		if head == '_\\/_' or head == '_/\\_':
			g = head.replace('_', '').join(rest_str)

		else:
			for ch in head:
				if ch == '_':
					g += next(rest_it)
				elif ch != '`':
					g += ch

		return g


def _actionParse(aprop):
	"""Convert an action to string removing the type annotation"""
	is_opaque = aprop.symbol().arity() > 0 and str(aprop.symbol()) == 'opaque'

	text = str(next(aprop.arguments()) if is_opaque else aprop)
	type_annot = text.find(').@Action@')

	clean_name = text[1:type_annot] if text[0] == '(' and type_annot > 0 else text

	return ['opaque', clean_name] if is_opaque else clean_name


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
	"""Convert the formula parsed by Maude into a pure Python list"""
	prop_sort, variable_sort, labels = extra_args

	if form.getSort() <= prop_sort:
		return ['Prop', str(form)]

	elif form.getSort() <= variable_sort:
		return ['Var', str(form)]

	elif str(form.symbol()) in ('<_>_', '`[_`]_'):
		symbol_args = list(form.arguments())
		args = [_actionSpecParse(symbol_args[0], labels), _formula2List(symbol_args[1], *extra_args)]

	else:
		args = [_formula2List(x, *extra_args) for x in form.arguments()]

	return [str(form.symbol())] + args


def _reviewOpaqueActions(form, rule_labels, opaque_labels):
	"""Check whether opaque is being misused for rule labels and mark strategy labels with opaque"""
	head = form[0]
	first_argument = 1

	if head in ('<_>_', '`[_`]_'):
		# The list is modified in place
		label_spec = form[1]

		for k in range(len(label_spec)):
			# Misused opaque wrappers are removed
			if label_spec[k][0] == 'opaque':
				if label_spec[k][1] not in opaque_labels:
					label_spec[k] = label_spec[k][1]
					usermsgs.print_warning(f'Label {label_spec[k]} does not refer to an opaque strategy. '
					                       'It will be interpreted as a rule label.')

			# Opaque labels are normalized to the opaque wrapper
			elif label_spec[k] not in rule_labels:
				label_spec[k] = ['opaque', label_spec[k]]

		first_argument = 2

	# Continue reviewing recursively
	if head not in ('Prop', 'Var'):
		for subform in form[first_argument:]:
			_reviewOpaqueActions(subform, rule_labels, opaque_labels)


def _int_orelse_float(term):
	"""Convert a term to an integer if possible, otherwise to a float"""

	value = float(term)
	return int(value) if value.is_integer() else value


def _boundSpecParse(bound):
	"""Parse a probability or steps bound"""

	symbol = str(bound.symbol())

	if symbol in ('<_', '<=_', '>_', '>=_'):
		return [symbol.replace('_', ''), _int_orelse_float(next(bound.arguments()))]

	else:
		return ['-'] + [_int_orelse_float(arg) for arg in bound.arguments()]


def _probFormula2List(form, prop_sort, aprops):
	"""Convert the probabilistic formula parsed by Maude into a pure Python list"""

	if form.getSort() <= prop_sort:
		aprops.add(str(form))
		return ['Prop', str(form)]

	elif str(form.symbol()) in ('<>__', '`[`]__', 'P__'):
		symbol_args = list(form.arguments())
		args = [_boundSpecParse(symbol_args[0]), _probFormula2List(symbol_args[1], prop_sort, aprops)]

	elif str(form.symbol()) in ('_U__', '_R__', '_W__'):
		symbol_args = list(form.arguments())
		args = [_boundSpecParse(symbol_args[1]),
		        _probFormula2List(symbol_args[0], prop_sort, aprops),
		        _probFormula2List(symbol_args[2], prop_sort, aprops)]

	else:
		args = [_probFormula2List(x, prop_sort, aprops) for x in form.arguments()]

	return [str(form.symbol())] + args


class BaseParser:
	"""Base class for temporal formulae parsers in umaudemc"""

	def __init__(self, parser_file, parser_module):
		# The module with the parsing module generator
		self.templog = None
		self.makeParserModule = None
		self.labels = None
		self.base_module = None

		self.get_templog(parser_file, parser_module)

	def is_ok(self):
		"""Whether the parser has been loaded successfully"""
		return self.makeParserModule is not None

	def set_module(self, module, metamodule=None):
		"""Set the module where formulae should be parsed"""

		# Build a list with the rule labels of the module (to allow
		# using them as edge labels in mu-calculus formulae)
		# (opaque strategies are not being considered)
		self.labels = [rl.getLabel() for rl in module.getRules()
		               if rl.getLabel() is not None]

		# Store the module as a metaterm in templog
		self.base_module = self.templog.parseTerm(f"upModule('{str(module)}, false)"
		                                          if metamodule is None else str(metamodule))

	def get_templog(self, parser_file, parser_module):
		"""Set up the parse module generator"""

		self.templog = maude.getModule(parser_module)

		# We load it if not already loaded
		if not self.templog:
			# The model-checker.maude file is only loaded once to prevent advisory messages
			# about modules being redefined (sload is not enough if a model-checker.maude file
			# is present in the directory of the input file)
			if not maude.getModule('STRATEGY-MODEL-CHECKER'):
				maude.load('model-checker')

			# Loads the temporal logic parser
			with get_resource_path(parser_file) as templog_path:
				if not maude.load(str(templog_path)):
					usermsgs.print_error(f'Error loading temporal logic parser ({parser_file}).')
					return False

			self.templog = maude.getModule(parser_module)

			if not self.templog:
				usermsgs.print_error(f'Error loading temporal logic parser ({parser_module} module).')
				return False

		module_kind = self.templog.findSort('Module').kind()
		qidlist_kind = self.templog.findSort('QidList').kind()

		if not module_kind or not qidlist_kind:
			usermsgs.print_error(f'Error loading temporal logic parser (bad {parser_module} module).')
			return False

		self.makeParserModule = self.templog.findSymbol('makeParserModule',
		                                                (module_kind, qidlist_kind, qidlist_kind),
		                                                module_kind)

		if not self.makeParserModule:
			usermsgs.print_error('Error loading temporal logic parser (missing makeParserModule operator).')

	def parse_aux(self, formula, label_list):
		"""Parse a temporal formulae using the makeParserModule procedure"""

		# Make a module where to parse the temporal formula
		parser_metamod = self.makeParserModule(
			self.base_module,
			self.templog.parseTerm('tokenize("{}")'.format(formula.replace('"', '\"'))),
			self.templog.parseTerm(label_list)
		)

		extmod = maude.downModule(parser_metamod)

		formula_kind = extmod.findSort('Formula').kind()

		# Parses the temporal formula and deduces its type
		parsed_formula = extmod.parseTerm(formula, formula_kind)
		# a copy can be made to preserve the original

		return parsed_formula, extmod


class Parser(BaseParser):
	"""
	Parser of any exact temporal formula supported by umaudemc

	Since the parsing module depends on the formula itself, there is no
	much we can catch between executions.
	"""

	PARSER_FILE = 'templog.maude'
	PARSER_MODULE = 'TEMPORAL-LOGIC-META'

	def __init__(self):
		super().__init__(self.PARSER_FILE, self.PARSER_MODULE)

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
		formulaElems = extmod.findSymbol('formulaElems', (formula_kind,), formelems_kind)
		formulaType = extmod.findSymbol('formulaType', (formelems_kind,), logicname_kind)

		if formulaElems is None or formulaType is None:
			usermsgs.print_error('Missing expected operators in the formula parsing module.'
			                     'Check the templog.maude file.')
			return None

		formulaLogic = formulaType(formulaElems(formula))
		formulaLogic.reduce()

		return str(formulaLogic)

	def parse(self, formula, opaques=()):
		"""Parse a temporal formula from LTL to mu-calculus"""

		# Makes a list of all rules to be parsed
		if len(self.labels) + len(opaques) > 0:
			all_labels = ' '.join(f"'{label}" for label in set(self.labels).union(set(opaques)))
		else:
			all_labels = '(nil).QidList'

		# Parse the temporal formula with Maude
		parsed_formula, extmod = self.parse_aux(formula, all_labels)

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

		# Convert the formula the internal list format and normalize μ-calculus actions
		formula_as_list = _formula2List(parsed_formula, prop_sort, mcvariable_sort, set(self.labels))
		_reviewOpaqueActions(formula_as_list, self.labels, opaques)

		return formula_as_list, logic


class ProbParser(BaseParser):
	""""Parser for probabilistic temporal formulae admitted by LTSmin"""

	PARSER_FILE = 'problog.maude'
	PARSER_MODULE = 'PROBABILISTIC-TEMPORAL-LOGIC-META'

	def __init__(self):
		super().__init__(self.PARSER_FILE, self.PARSER_MODULE)

	def parse(self, formula):
		"""Parse a probabilistic temporal formula"""

		# Labels are not used in probabilistic formulae for the moment
		all_labels = '(nil).QidList'

		# Parse the temporal formula using Maude
		parsed_formula, extmod = self.parse_aux(formula, all_labels)

		if parsed_formula is None:
			usermsgs.print_error('The given probabilistic temporal formula cannot be parsed.')
			return None, None

		prop_sort = extmod.findSort('Prop')

		# Reduce the formula just in case it contains defined functions
		parsed_formula.reduce()

		# Convert the formula to the internal list-based format
		# while collecting its atomic propositions
		aprops = set()
		formula_as_list = _probFormula2List(parsed_formula, prop_sort, aprops)

		return formula_as_list, aprops
