#
# Ad-hoc extension
#

import re

from docutils import nodes, utils
from docutils.utils.math import MathError
from docutils.utils.math.latex2mathml import tex2mathml
# from docutils.utils.math.tex2mathml_extern import pandoc

# Regular expression for math escapes
MATH_RE = re.compile(r'\$([^\$]+)\$')


class codemath(nodes.Inline, nodes.Element):
	pass


def codemath_role(role, rawtext, text, lineno, inliner, option={}, content=[]):
	code = utils.unescape(text, restore_backslashes=True)

	return [codemath(code=code)], []


def html_visit_codemath(self, node):
	code = MATH_RE.sub(lambda m: tex2mathml(m.group(1)), node.get('code'))
	self.body.append(f'<code class="docutils literal notranslate"><span class="pre">{code}</span></code>')
	raise nodes.SkipNode


def latex_visit_codemath(self, node):
	self.body.append(f'\\texttt{{{node.get("code")}}}')
	raise nodes.SkipNode


def html_visit_math(self, node):
	self.body.append(self.starttag(node, 'span', '', CLASS='math notranslate nohighlight'))
	try:
		self.body.append(f'{tex2mathml(node.astext())}</span>')
	except MathError as me:
		self.document.reporter.error(str(me))
	raise nodes.SkipNode


def html_visit_displaymath(self, node):
	self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
	try:
		self.body.append(f'{tex2mathml(node.astext(), as_block=True)}</div>')
	except MathError as me:
		self.document.reporter.error(str(me))
	raise nodes.SkipNode


def setup(app):
	"""Setup custom directives"""

	# codemath directive (for code with math escapes)
	app.add_role('codemath', codemath_role)
	app.add_node(codemath,
		html=(html_visit_codemath, None),
		latex=(latex_visit_codemath, None)
	)

	# Math renderer to MathML
	app.add_html_math_renderer('mathml',
		inline_renderers=(html_visit_math, None),
		block_renderers=(html_visit_displaymath, None),
	)

	return {
		'version': '1.0',
		'parallel_read_safe': True,
		'parallel_write_safe': True,
	}
