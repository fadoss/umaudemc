#
# Parser and compiler for the QuaTEx language
#

import ast

from . import usermsgs


class QuaTExProgram:
	"""Compiled QuaTEx program"""

	def __init__(self, slots, varnames, ndefs, qinfo):
		# Slots of the QuaTEx program
		# (contain code objects for the definitions and queries)
		self.slots = slots
		# Variable name tuples for every definition (indexed as in slots)
		self.varnames = varnames
		# Number of definitions (unused definitions are not counted)
		self.ndefs = ndefs
		# Number of queries
		self.nqueries = len(slots) - ndefs

		# Query information (line, column, and parameters)
		self.query_locations = qinfo


class QuaTExLexer:
	"""Lexer for QuaTEx"""

	LT_NAME = 0
	LT_NUMBER = 1
	LT_STRING = 2
	LT_OTHER = 3

	def __init__(self, source, buffer_size=512):
		# Input stream (file-like object)
		self.source = source
		# We read in chunks of buffer_size
		self.buffer_size = buffer_size
		self.buffer = self.source.read(self.buffer_size)
		# Whether there are no more tokens available
		self.exhausted = len(self.buffer) == 0
		# Index of the current character in the buffer
		self.index = 0
		# Lexeme type and content
		self.ltype = None
		self.lexeme = ''

		# Current line and column numbers
		self.line = 1
		self.column = 1

		# Line and column numbers where the token starts
		self.sline = 1
		self.scolumn = 1

	def _next(self):
		"""Advance to the next character in the input"""

		if self._current == '\n':
			self.line += 1
			self.column = 1
		else:
			self.column += 1

		# We accumulate the input in the lexeme field
		self.lexeme += self._current
		self.index += 1

		# The buffer is exhausted
		if self.index >= len(self.buffer):
			if len(self.buffer) == self.buffer_size:
				self.buffer = self.source.read(self.buffer_size)
				self.exhausted = len(self.buffer) == 0
				self.index = 0
			else:
				self.exhausted = True

	def _peek(self):
		"""Peek the next character without consuming it"""

		if self.index + 1 < len(self.buffer):
			return self.buffer[self.index + 1]

		elif not self.exhausted:
			self.buffer = self._current + self.source.read(self.buffer_size - 1)
			self.index = 0

			if 1 < len(self.buffer):
				return self.buffer[1]

	@property
	def _current(self):
		"""Current character"""

		return self.buffer[self.index]

	def _ignore_line(self):
		"""Ignore the rest of the line"""

		while not self.exhausted and self._current != '\n':
			if self.index + 1 >= len(self.buffer):
				self._next()  # next is called to refill the buffer
			else:
				self.index += 1  # we avoid appeding to the lexeme...

		# Skip the newline
		if not self.exhausted:
			self._next()

	def _capture(self, p):
		"""Capture text while a predicate holds"""

		while not self.exhausted and p(self._current):
			self._next()

	def _capture_string(self):
		"""Capture a quoted string"""

		escaped = False

		# Skip the starting quotation mark
		self._next()

		while not self.exhausted and (escaped or self._current != '"'):
			if escaped:
				escaped = False

			elif self._current == '\\':
				escaped = True

			self._next()

		# Skip the final quotation mark
		self._next()

	def _begin_lexeme(self):
		"""Start a lexeme"""

		self.lexeme = ''
		self.sline = self.line
		self.scolumn = self.column

	def get_token(self):
		"""Get the next token from the stream"""

		# Ignore whitespace and comments
		while not self.exhausted:
			if self._current.isspace():
				self._next()
			elif self._current == '/' and self._peek() == '/':
				self._ignore_line()
			else:
				break

		if self.exhausted:
			return None

		self._begin_lexeme()
		c = self._current

		if c == '"':
			self.ltype = self.LT_STRING
			self._capture_string()

		elif c.isalpha():
			self.ltype = self.LT_NAME
			self._capture(str.isalnum)

		elif c.isdecimal():
			self.ltype = self.LT_NUMBER

			self._capture(str.isdecimal)

			# Check for the decimal separator
			if not self.exhausted and self._current == '.':
				self._next()
				self._capture(str.isdecimal)

			# Check for the exponent
			if not self.exhausted and self._current.lower() == 'e':
				self._next()

				if not self.exhausted and self._current in '+-':
					self._next()

				self._capture(str.isdecimal)

		else:
			self.ltype = self.LT_OTHER

			# Operators ==, !=, <=, >=, &&, and || are single tokens
			if (c in '=!<>' and self._peek() == '=' or
			    c == '&' and self._peek() == '&' or
			    c == '|' and self._peek() == '|'):
				self._next()

			self._next()

		return self.lexeme


class QuaTExParser:
	"""Parser for QuaTEx"""

	PS_IFC = 0  # condition
	PS_IFT = 1  # true branch
	PS_IFF = 2  # negative branch
	PS_PAREN = 3  # parenthesis
	PS_ARITH = 4  # completing an arithmetic expression
	PS_CALLARGS = 5  # call arguments

	# Binary operator and their precedences (as in C)
	BINARY_OPS = ('+', '-', '*', '/', '%', '&&', '||', '==', '!=', '<', '<=', '>', '>=')
	BINOPS_PREC = (4, 4, 3, 3, 3, 11, 12, 7, 7, 6, 6, 6, 6)
	BINOPS_AST = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.And, ast.Or, ast.Eq,
	              ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
	BINOPS_CMP = (0, ) * 5 + (1, ) * 2 + (2, ) * 6
	# Unary operator and its precedence (as in C)
	UNARY_OPS = ('!', )
	UNARY_AST = (ast.Not, )

	def __init__(self, source, filename='<stdin>'):
		self.lexer = QuaTExLexer(source)
		# Filename is only used for diagnostics
		self.filename = filename

		# Parameters of the current function
		self.fvars = []
		# Whether the variables that may occur
		# in an expression are known
		self.known_vars = True

		# State stack for parsing expressions
		self.stack = []
		self.prec = 0

		# Whether parsing errors have been encountered
		self.ok = True

		# Compilation slot indices for each number
		self.fslots = {}
		self.calls = []
		self.observations = []
		self.queries = []
		self.defs = []

	def _eprint(self, msg, line=None, column=None):
		"""Print an error message with line information"""

		usermsgs.print_error_loc(self.filename,
		                         line or self.lexer.sline,
		                         column or self.lexer.scolumn,
		                         msg)

	def _expect(self, *text):
		"""Check whether the given expected tokens are present"""

		for etoken in text:
			atoken = self.lexer.get_token()

			if atoken is None:
				self._eprint(f'unexpected end of file where "{etoken}" is required.')
				return False

			if etoken != atoken:
				self._eprint(f'unexpected token "{atoken}" where "{etoken}" is required.')
				return False

		return True

	def _in_state(self, state):
		"""Check whether the parser is in the given state"""

		return self.stack and self.stack[-1] == state

	def _do_binop(self, op_index, left, right):
		"""Build a binary operator in the AST"""

		op_type = self.BINOPS_CMP[op_index]

		if op_type == 2:
			return ast.Compare(left, [self.BINOPS_AST[op_index]()], [right])

		if op_type == 1:
			return ast.BoolOp(self.BINOPS_AST[op_index](), [left, right])

		return ast.BinOp(left, self.BINOPS_AST[op_index](), right)

	def _do_unaryop(self, op_index, argument):
		"""Build a unary operator in the AST"""

		return ast.UnaryOp(self.UNARY_AST[op_index](), operand=argument)

	def _parse_parameter(self):
		"""Parse the parameter specification of a parametric query"""

		if not self._expect(','):
			return (None, ) * 3

		var_name = self.lexer.get_token()

		if self.lexer.ltype != self.lexer.LT_NAME:
			self._eprint(f'unexpected token "{var_name}" where a variable name is required.')
			return (None, ) * 3

		# Parameter specification (name, initial value, step, last value)
		spec = [var_name]

		# Parse the initial value, step, and last value
		for _ in range(3):
			if not self._expect(','):
				return (None, ) * 3

			token = self.lexer.get_token()

			if self.lexer.ltype != self.lexer.LT_NUMBER:
				self._eprint(f'unexpected token "{token}" where a number is required.')
				return (None, ) * 3

			spec.append(float(token))

		# Check the closing parenthesis
		if not self._expect(')'):
			return (None, ) * 3

		# Check whether the variables in the expressions are the parameter
		for var, line, column in self.fvars:
			if var != var_name:
				self._eprint(f'unknown variable "{var}".', line=line, column=column)
				self.ok = False

		self.fvars.clear()

		return tuple(spec)

	def _parse_expr(self, end_token, inside_def=False):
		"""Parse an expression"""

		# Current expression
		current = None
		# Number of nested conditions in the current position
		inside_cond = 0
		# Information about the last call
		call_name = None
		call_line = None
		call_column = None
		# Whether a call is preceded by next
		inside_next = False
		# Stack of arguments of pending expressions
		arg_stack = []

		token = self.lexer.get_token()

		while True:
			# Finish arithmetic/logical expressions
			while self._in_state(self.PS_ARITH) and current and token not in self.BINARY_OPS:
				self.stack.pop()
				left, op_index = arg_stack[-1]
				current = self._do_binop(op_index, left, current) if left else self._do_unaryop(op_index, current)
				arg_stack.pop()

			if token == end_token:
				if self.stack or not current:
					self._eprint(f'unexpected end of expression when "{end_token}" is found.')
					return None

				return current

			elif token == 'if':
				if current:
					self._eprint('misplaced "if" keyword.')
					return None

				self.stack.append(self.PS_IFC)
				arg_stack.append([])
				inside_cond += 1

			elif token == 'then':
				if current and self._in_state(self.PS_IFC):
					self.stack[-1] = self.PS_IFT
					arg_stack[-1].append(current)
					current = None
					inside_cond -= 1
				else:
					self._eprint('misplaced "then" keyword.')
					return None

			elif token == 'else':
				if current and self._in_state(self.PS_IFT):
					self.stack[-1] = self.PS_IFF
					arg_stack[-1].append(current)
					current = None
				else:
					self._eprint('misplaced "else" keyword.')
					return None

			elif token == 'fi':
				if current and self._in_state(self.PS_IFF):
					arg_stack[-1].append(current)
					current = ast.IfExp(*arg_stack[-1])
					arg_stack.pop()
					self.stack.pop()
				else:
					self._eprint('misplaced "fi" keyword.')
					return None

			elif token == '#':
				if current:
					self._eprint('misplaced next # operator.')
					return None

				if inside_cond or call_name:
					self._eprint('the next operator # cannot be used in conditions or call arguments.')
					self.ok = False

				# A function call should follow
				token = self.lexer.get_token()
				line, column = self.lexer.sline, self.lexer.scolumn

				if self.lexer.ltype != self.lexer.LT_NAME or self.lexer.get_token() != '(':
					self._eprint(f'the next # operator must be followed by a function call, but "{token}" is found instead.')
					return None

				self.stack.append(self.PS_CALLARGS)
				arg_stack.append([])
				inside_next = True
				call_name, call_line, call_column = token, line, column

			elif token == ',':
				if current and self._in_state(self.PS_CALLARGS):
					arg_stack[-1].append(current)
					current = None
				else:
					self._eprint('misplaced comma.')
					return None

			elif token == '(':
				if current:
					self._eprint(f'misplaced parenthesis.')
					return None

				self.stack.append(self.PS_PAREN)

			elif token == ')':
				# Closing a function call
				if self._in_state(self.PS_CALLARGS):
					self.stack.pop()

					# The arguments of the call (except the last one, if any)
					args = arg_stack.pop()

					if current:
						args.append(current)
					elif args:
						self._eprint(f'argument missing after a comma in a call to "{call_name}".')
						self.ok = False

					slot = self.fslots.setdefault(call_name, len(self.fslots))
					current = ast.Tuple([ast.Constant(inside_next), ast.Constant(slot), *args],
					                    ast.Load(), custom_loc=(call_line, call_column))
					self.calls.append((call_name, call_line, call_column, len(args)))
					inside_next = False
					call_name = None

				# Closing an organizational parenthesis
				elif current and self._in_state(self.PS_PAREN):
					self.stack.pop()

				else:
					self._eprint(f'unexpected parenthesis.')
					return None

			elif self.lexer.ltype == self.lexer.LT_NAME:
				if current:
					self._eprint(f'unexpected identifier "{token}".')
					return None

				# Get the next token to see whether it is a function call
				# and s.rval or simply a variable
				line, column = self.lexer.sline, self.lexer.scolumn
				next_token = self.lexer.get_token()

				# A call to s.reval
				if token == 's' and next_token == '.':
					if not self._expect('rval', '('):
						return None

					token = self.lexer.get_token()
					ltype = self.lexer.ltype

					if ltype not in (self.lexer.LT_NAME, self.lexer.LT_STRING, self.lexer.LT_NUMBER):
						self._eprint(f's.rval only admits string literals and variables, but "{token}" is found.')
						return None

					if not self._expect(')'):
						return None

					if ltype == self.lexer.LT_NAME:
						argument = ast.Name(token, ast.Load())
					elif ltype == self.lexer.LT_NUMBER:
						argument = ast.Constant(int(token))
					else:
						token = token[1:-1]
						self.observations.append(token)
						argument = ast.Constant(token)

					current = ast.Call(ast.Name('rval', ast.Load()), [argument], [])

				# Function call
				elif next_token == '(':
					if inside_cond or call_name:
						self._eprint(f'"{token}" is called in a condition or call argument, but this is not allowed.')
						return None

					self.stack.append(self.PS_CALLARGS)
					arg_stack.append([])
					call_name = token
					call_line, call_column = line, column

				# Simply a variable
				else:
					if not self.known_vars:
						self.fvars.append((token, line, column))

					elif token not in self.fvars:
						self._eprint(f'unknown variable "{token}".', line=line, column=column)
						self.ok = False

					current = ast.Name(token, ast.Load())

					# We continue with the peeked token
					token = next_token
					continue

			elif self.lexer.ltype == self.lexer.LT_NUMBER:
				if current:
					self._eprint(f'unexpected number "{token}".')

				current = ast.Constant(float(token))

			elif self.lexer.ltype == self.lexer.LT_STRING:
				if not call_name:
					self._eprint(f'strings like {token} can only appear in rval and function arguments.')
					return None

				current = ast.Constant(token[1:-1])

			elif self.lexer.ltype == self.lexer.LT_OTHER:
				if token in self.BINARY_OPS:
					# The left operand of token is missing
					if not current:
						self._eprint(f'the left operand for "{token}" is missing.')
						return None

					op_index = self.BINARY_OPS.index(token)

					# If we are continuing an arithmetic expression
					if not self._in_state(self.PS_ARITH):
						self.stack.append(self.PS_ARITH)
						arg_stack.append((current, op_index))
						current = None

					else:
						left, last_op_index = arg_stack[-1]

						# The precedence of the previous operand is greater
						# or equal than that of the new one
						if self.BINOPS_PREC[last_op_index] <= self.BINOPS_PREC[op_index]:
							arg_stack[-1] = self._do_binop(last_op_index, left, current), op_index
							current = None
						else:
							self.stack.append(self.PS_ARITH)
							arg_stack.append((current, op_index))
							current = None

				elif token in self.UNARY_OPS:
					# There should be no left operand
					if current:
						self._eprint(f'misplaced "{token}" operator.')
						return None

					op_index = self.UNARY_OPS.index(token)

					# Parse the argument
					self.stack.append(self.PS_ARITH)
					arg_stack.append((None, op_index))

				else:
					self._eprint(f'unexpected token "{token}".')
					return None

			token = self.lexer.get_token()

	def _parse_toplevel(self):
		"""Parse the top level of a QuaTeX file"""

		token = self.lexer.get_token()

		while not self.lexer.exhausted:

			# Any top level statement starts with eval or other identifier
			if self.lexer.ltype != self.lexer.LT_NAME:
				self._eprint(f'unexpected token "{token}" at the top level.')
				return False

			# Query -- eval E [ <expr> ] ;
			if token == 'eval':
				# The query location is kept for future reference
				line, column = self.lexer.sline, self.lexer.scolumn

				# Either 'E' for simple or 'parametric' for parametric queries
				token = self.lexer.get_token()
				parameter = token == 'parametric'

				if not parameter and token != 'E':
					self._eprint(f'unexpected token "{token}" where "E" or "parametric" is required.')
					return False

				if not (self._expect('(', 'E', '[') if parameter else self._expect('[')):
					return False

				# When parsing parameterized expressions, the parameter name is not
				# known in advance (appears later in the syntax)
				if parameter:
					self.known_vars = False

				expr = self._parse_expr(']')

				self.known_vars = True

				# Parse parameter specification in parametric queries
				parameter = self._parse_parameter() if parameter else None

				if not expr or not self._expect(';'):
					return False

				# Ignore parameterized expressions with empty range
				if parameter and parameter[1] > parameter[3]:
					usermsgs.print_warning_loc(self.filename, line, column,
					                           'ignoring parametric query with empty range.')
				else:
					self.queries.append((line, column, expr, parameter))

			# Function definition -- <name> ( <args> ) = <expr> ;
			else:
				# The function name
				fname, line, column = token, self.lexer.sline, self.lexer.scolumn

				if not self._expect('('):
					return False

				token = self.lexer.get_token()
				more_args = token != ')'

				# Get the list of variables
				while more_args:
					if self.lexer.ltype != self.lexer.LT_NAME:
						self._eprint(f'unexpected token "{token}" where a variable name is required.')
						return None

					self.fvars.append(token)

					# Look for the comma or closing parenthesis
					token = self.lexer.get_token()

					if token == ')':
						more_args = False
					elif token == ',':
						token = self.lexer.get_token()
					else:
						self._eprint(f'unexpected token "{token}" where "," or ")" is expected.')
						return None

				if not self._expect('='):
					return False

				expr = self._parse_expr(';', inside_def=True)

				if not expr:
					return False

				self.defs.append((fname, line, column, tuple(self.fvars), expr))
				self.fvars.clear()

			token = self.lexer.get_token()

		return self.ok

	def _check_tail(self, expr):
		"""Check whether all function calls are tail"""

		pending = [(expr, True)]

		while pending:
			expr, tail_pos = pending.pop()

			if isinstance(expr, ast.Tuple) and not tail_pos:
				self._eprint('non-tail calls are not allowed.',
				             line=expr.custom_loc[0], column=expr.custom_loc[1])
				return False

			elif isinstance(expr, ast.UnaryOp):
				pending.append((expr.operand, False))

			elif isinstance(expr, ast.BinOp):
				pending.append((expr.right, False))
				pending.append((expr.left, False))

			elif isinstance(expr, ast.IfExp):
				pending.append((expr.orelse, tail_pos))
				pending.append((expr.body, tail_pos))

		return True

	def parse(self):
		"""Parse a QuaTeX file"""

		# Parse the file
		if not self._parse_toplevel():
			return None

		# Some additional checks
		arities, ok = {}, True

		# Check functions are only defined once
		for name, line, column, args, _ in self.defs:
			if name in arities:
				self._eprint(f'multiply defined function "{name}".', line=line, column=column)
				ok = False
			else:
				arities[name] = len(args)

		# Check whether all called are well-defined
		for name, line, column, arity in self.calls:
			def_arity = arities.get(name)

			if def_arity is None:
				self._eprint(f'call to undefined function "{name}".',
				             line=line, column=column)
				ok = False

			elif arity != def_arity:
				self._eprint(f'wrong number of arguments in a call to "{name}" ({arity} given, but {def_arity} expected).',
				             line=line, column=column)
				ok = False

		# Check all calls are tail in expression
		for name, line, column, _, expr in self.defs:
			if not self._check_tail(expr):
				ok = False

		for line, column, expr, _ in self.queries:
			if not self._check_tail(expr):
				ok = False

		if not ok:
			return None

		# Compile the program as an array of slots according to the indices in fslots
		used_defs = len(self.fslots)

		slots = [None] * (used_defs + len(self.queries))
		varnames = [None] * used_defs

		for name, line, column, args, body in self.defs:
			index = self.fslots.get(name)

			if index is None:
				continue

			varnames[index] = args

			try:
				body = ast.Expression(body)
				ast.fix_missing_locations(body)
				slots[index] = compile(body, filename=name, mode='eval')

			except TypeError:
				self._eprint(f'the definition of "{name}" cannot be compiled.',
				             line=line, column=column)
				ok = False

		for k, (line, column, expr, _) in enumerate(self.queries):
			try:
				expr = ast.Expression(expr)
				ast.fix_missing_locations(expr)
				slots[used_defs + k] = compile(expr, filename=f'query{line}:{column}', mode='eval')

			except TypeError:
				self._eprint('this query cannot cannot be compiled.',
				             line=line, column=column)
				ok = False

		if not ok:
			return None

		return QuaTExProgram(slots, varnames, len(self.fslots),
		                     tuple((line, column, params) for line, column, _, params in self.queries))


def parse_quatex(input_file, filename='<string>'):
	"""Parse a QuaTEx formula"""

	# Load, parse, and compile the QuaTEx file
	return QuaTExParser(input_file, filename=filename).parse()
