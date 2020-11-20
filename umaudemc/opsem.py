#
# Extensible small-step operational semantics of the Maude strategy language
#

import maude

from .resources import get_resource_path
from . import usermsgs


#
# View to instantiate the small-step operational semantics with a given
# module, and module where all required components are put all together.
#
# Instantiation and generation of views and modules are not supported by the
# Maude bindings, but we can always make Maude load views and modules as/ text.
#
INSTANTIATION = '''view OpSem-{instance_label} from MODULE to META-LEVEL is
	op M to term {target_module} .
endv

smod OPSEM-MAIN-{instance_label} is
	protecting NOP-PREDS{{OpSem-{instance_label}}} .
	protecting {semantics_module}{{OpSem-{instance_label}}} .
	protecting LEXICAL .
	including STRATEGY-MODEL-CHECKER .
endsm'''


class OpSemInstance:
	"""Instance of the operational semantics for a module"""

	# Default module name for the semantics
	SEMANTICS_MODULE = 'NOP-SEMANTICS'

	def __init__(self, osmod, targetmod):
		# Module of the operational semantics
		self.osmod = osmod
		# Module where terms are rewritten
		self.targetmod = targetmod

		# Attribute to be initialized later
		self.term_sort = None
		self.strategy_sort = None
		self.cterm = None
		self.stack_state = None

		self._load_values()

	def _load_values(self):
		"""Load sorts and operator required to extract information from the semantics"""

		# Sort required to find symbols in the semantics
		expart_kind = self.osmod.findSort('ExStatePart').kind()
		exstate_kind = self.osmod.findSort('ExState').kind()
		self.term_sort = self.osmod.findSort('Term')
		term_kind = self.term_sort.kind()
		self.strategy_sort = self.osmod.findSort('Strategy')
		strategy_kind = self.strategy_sort.kind()

		# Symbols for decomposing the semantic states
		self.cterm = self.osmod.findSymbol('cterm', [exstate_kind], term_kind)
		self.stack_state = self.osmod.findSymbol('_@_', [expart_kind, strategy_kind], exstate_kind)

	@classmethod
	def make_instance(cls, module, metamodule=None, semantics_module=None):
		"""Make an instance of the semantics for the given problem"""

		# Use the default semantics module if not specified
		if semantics_module is None:
			semantics_module = cls.SEMANTICS_MODULE

		# Load the opsem.maude file if not already done
		if maude.getModule(semantics_module) is None:
			with get_resource_path('opsem.maude') as templog_path:
				if not maude.load(str(templog_path)):
					usermsgs.print_error('Error loading the small-step operational semantics'
					                     'of the strategy language (opsem.maude).')
					return None

		# Instantiate the semantics for the given module
		maude.input(INSTANTIATION.format(
			instance_label=hash(metamodule) if metamodule is not None else str(module),
			target_module=metamodule if metamodule is not None else f"upModule('{module}, true)",
			semantics_module=semantics_module
		))
		osmod = maude.getCurrentModule()

		if osmod is None:
			usermsgs.print_error('Error instantiating the small-step operational semantics.')
			return None

		return cls(osmod, module)

	def make_initial_term(self, term, strategy):
		"""Make the initial term of the operational semantics"""

		# Reduce the initial term
		term.reduce()

		# Raise them to the metalevel of the opsem module
		term_meta = self.osmod.upTerm(term)
		strategy_meta = self.osmod.upStrategy(strategy)

		if term_meta is None or strategy_meta is None:
			return None

		# Construct the t @ s initial state
		return self.stack_state(term_meta, strategy_meta)

	def make_graph(self, initial, strategy, opaques=()):
		"""Get the strategy for the strategy-controlled rewriting"""

		# Build the initial term
		t = self.make_initial_term(initial, strategy)

		if opaques:
			opaque_set = ' ; '.join(["'" + name for name in opaques])
			s = self.osmod.parseStrategy(f'opsemo({opaque_set})')
		else:
			s = self.osmod.parseStrategy('opsem')

		return maude.StrategyRewriteGraph(t, s, ['->>o' if opaques else '->>'])

	def get_cterm(self, xst):
		"""Get the current term represented by the execution state"""

		t = self.cterm(xst)
		t.reduce()
		return self.targetmod.downTerm(t)


class OpSemKleeneInstance(OpSemInstance):
	"""Instance of the operational semantics with support for the Kleene semantics of the iteration"""

	# Default module name for the semantics
	SEMANTICS_MODULE = 'NOP-KLEENE-SEMANTICS'

	def __init__(self, targetmod, osmod):
		super().__init__(targetmod, osmod)

		self.wrap = None
		self.none_tag = None
		self.tag_list = None

		self._load_values()

	def _load_values(self):
		super()._load_values()

		exstate_kind = self.osmod.findSort('ExState').kind()
		tags_kind = self.osmod.findSort('ActionTags').kind()
		wstate_kind = self.osmod.findSort('WraptState').kind()

		self.wrap = self.osmod.findSymbol('wrap', [exstate_kind, tags_kind], wstate_kind)
		self.none_tag = self.osmod.parseTerm('none', tags_kind)
		self.tag_list = self.osmod.findSymbol('__', [tags_kind] * 2, tags_kind)

	def make_initial_term(self, term, strategy):
		semterm = super().make_initial_term(term, strategy)

		return self.wrap(semterm, self.none_tag)

	def get_cterm(self, xst):
		"""Get the current term in the execution state"""

		return super().get_cterm(next(xst.arguments()))

	def get_exstate(self, wst):
		"""Get the execution state in the wrapped state"""

		return next(wst.arguments())

	def get_tags(self, xst):
		"""Get the tags associated to an execution state"""

		# Ignore the first argument and get the second one
		it = xst.arguments()
		next(it)
		return next(it)

	def extract_tags(self, xst):
		"""Extract iteration enter/leave tags as a Python list"""

		tags_term = self.get_tags(xst)

		if tags_term.symbol() == self.tag_list:
			tags = list(tags_term.arguments())
		else:
			tags = [tags_term]

		return [(next(tag.arguments()), str(tag.symbol()) == 'enter') for tag in tags if
		        str(tag.symbol()) in ['enter', 'leave']]
