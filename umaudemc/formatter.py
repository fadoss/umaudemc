#
# Formatter functions for state and edge labels in counterexamples and graphs
#

import re

import maude


def apply_state_format(graph, index, sformat, terms, use_term=False, use_strat=False):
	"""
	State label generator using a prebuilt format string.

	:param graph: Rewriting graph the state belongs to.
	:type graph: Maude rewriting graph
	:param index: Index of the state in the graph.
	:type index: int
	:param sformat: Prebuilt format string.
	:type sformat: str
	:param terms: Term patterns to appear in the format string.
	:type terms: list of string
	:param use_term: Whether the term is actually used (only for effiency).
	:type use_term: bool
	:param use_strat: Whether the strategy is actually used (only for effiency).
	:type use_strat: bool
	:returns: Formatted state label.
	:rtype: string
	"""

	args = {'index': index}
	module = graph.getStateTerm(0).symbol().getModule()

	if use_term or terms:
		reusable_term = graph.getStateTerm(index)
		args['term'] = str(reusable_term)
		reusable_term = reusable_term.prettyPrint(maude.PRINT_WITH_PARENS)
	if use_strat:
		args['strat'] = str(graph.getStateStrategy(index)) if graph.strategyControlled else ''

	for i, preterm in enumerate(terms):
		# The term replaced in the term pattern cannot be the formatted
		# term, but a surely unambiguous term representation.
		term = module.parseTerm(preterm.replace('%t', reusable_term))
		term.reduce()
		args[f't{i}'] = str(term)

	return sformat.format(**args)


def parse_state_format(sformat, strategy):
	"""Parse a state-label format specification and returns a function
	that given the graph and a state index produces its label"""
	terms, fstring = [], ''
	i, j = 0, 0

	# The sformat string is read and the pieces of text between brackets
	# are collected in the terms list. fstring have this fragments replaced
	# by a format pattern {tn} where n is the index of the term in the list.

	while i < len(sformat):
		if sformat[i] == '{':
			fstring += sformat[j:i] + '{t' + str(len(terms)) + '}'
			j, depth = i+1, 1
			while depth > 0 and j < len(sformat):
				if sformat[j] == '{':
					depth += 1
				elif sformat[j] == '}':
					depth -= 1
				j = j + 1
			terms.append(sformat[i+1:j-1])
			i = j
		else:
			i = i + 1

	fstring += sformat[j:]

	# Spare term, strategy and index templates are replaced too

	fstring, tused = re.subn(r'%(.\d+)?t', r'{term:\1}', fstring)

	if strategy:
		fstring, sused = re.subn(r'%(.\d+)?s', r'{strat:\1}', fstring)
	else:
		sused = 0

	fstring = re.sub(r'%(.\d+)?i', r'{index}', fstring)

	return (lambda graph, index: apply_state_format(graph, index, fstring, terms,
	                                                tused > 0, sused > 0))


def apply_edge_format(graph, origin, dest, eformat):
	"""
	Edge label generator using a prebuilt format string.

	:param graph: Rewriting graph the transition belong to.
	:type graph: Maude rewriting graph.
	:param origin: Index of the origin state within the graph.
	:type origin: int
	:param dest: Index of the destination state within the graph.
	:type dest: int
	:param eformat: Prebuilt format string.
	:type eformat: str
	:returns: Formatted edge label.
	:rtype str
	"""

	stmt = graph.getRule(origin, dest)
	label = stmt.getLabel()
	line = stmt.getLineNumber()
	opaque = ''

	if label is None:
		label = ''

	return eformat.format(stmt=stmt, label=label, line=line, opaque=opaque)


def apply_edge_format_strat(graph, origin, dest, eformat):
	"""Edge label generator for strategy-controlled using a prebuilt format string"""
	trans = graph.getTransition(origin, dest)
	opaque, label, stmt, line = '', '', '', ''

	if trans.getType() == maude.StrategyRewriteGraph.SOLUTION:
		stmt = label = 'solution'

	elif trans.getType() == maude.StrategyRewriteGraph.RULE_APPLICATION:
		stmt = trans.getRule()
		label = stmt.getLabel()
		line = stmt.getLineNumber()

	elif trans.getType() == maude.StrategyRewriteGraph.OPAQUE_STRATEGY:
		stmt = trans.getStrategy()
		label = stmt.getName()
		line = stmt.getLineNumber()
		opaque = 'opaque'

	return eformat.format(stmt=stmt, label=label, line=line, opaque=opaque)


def parse_edge_format(eformat, strategy):
	"""Parse edge format specification and returns a function that
	given a graph and two state indices generates their transition label"""

	eformat = eformat.replace('{', '{{').replace('}', '}}')

	eformat = re.sub(r'%(.\d+)?s', r'{stmt:\1}', eformat)
	eformat = re.sub(r'%(.\d+)?l', r'{label:\1}', eformat)
	eformat = re.sub(r'%(.\d+)?o', r'{opaque:\1}', eformat)
	eformat = re.sub(r'%(.\d+)?n', r'{line:\1}', eformat)

	if strategy:
		return lambda graph, origin, dest: apply_edge_format_strat(graph, origin, dest, eformat)
	else:
		return lambda graph, origin, dest: apply_edge_format(graph, origin, dest, eformat)


#
# The following are printing function for fixed format,
# which are used as defaults.
#

def print_term(graph, index):
	"""Default state-label printing function"""
	return graph.getStateTerm(index)


def print_transition(graph, origin, dest):
	"""Default edge-label printing function"""
	return graph.getRule(origin, dest)


def print_transition_strat(graph, origin, dest):
	"""Default edge-label printing function with strategies"""
	trans = graph.getTransition(origin, dest)
	return {
		maude.StrategyRewriteGraph.RULE_APPLICATION	: trans.getRule(),
		maude.StrategyRewriteGraph.OPAQUE_STRATEGY	: trans.getStrategy(),
		maude.StrategyRewriteGraph.SOLUTION		: ''
	}[trans.getType()]


def print_transition_label(graph, origin, dest):
	"""Alternative edge-label printing function (only rule label)"""
	return graph.getRule(origin, dest).getLabel()


def print_transition_strat_label(graph, origin, dest):
	"""Alternative edge-label printing function with strategies (only rule/strategy label)"""
	trans = graph.getTransition(origin, dest)
	ttype = trans.getType()

	if ttype == maude.StrategyRewriteGraph.RULE_APPLICATION:
		return trans.getRule().getLabel()
	elif ttype == maude.StrategyRewriteGraph.OPAQUE_STRATEGY:
		return trans.getStrategy().getName()
	else:
		return ''


def get_formatters(sspec, espec, withStrategy, only_labels=False):
	"""Get state and edge label formatter functions from format specifications"""
	edefault = (print_transition_strat_label if withStrategy else print_transition_label) if only_labels \
		else (print_transition_strat if withStrategy else print_transition)

	slabel = parse_state_format(sspec, withStrategy) if sspec is not None else print_term
	elabel = parse_edge_format(espec, withStrategy) if espec is not None else edefault

	return slabel, elabel
