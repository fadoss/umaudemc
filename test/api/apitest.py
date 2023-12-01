#
# Unit test for the API
#

import io
import unittest
import sys

import maude
import umaudemc.api as api

# Init Maude and load its files
maude.init(advise=False)
maude.load('../check/vending')


class APITest(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.m = maude.getModule('VENDING-MACHINE-CHECK')
		self.t = self.m.parseTerm('initial')
		self.a = self.m.parseStrategy('put1 ; apple | put1 ; put1 ; cake')
		self.b = self.m.parseStrategy('put1 ; (apple | put1 ; cake)')

		self.model = api.MaudeModel(self.t)
		self.smodel = api.MaudeModel(self.t, strategy=self.a)

	def test_graph(self):
		"""Graph generation"""

		with io.StringIO() as out:
			self.model.print_graph(out)
			text = out.getvalue()
			self.assertTrue(text.startswith('digraph {'))
			self.assertTrue('[label="e e [empty]"]' in text)
			self.assertTrue('label="rl e O:Soup [I:Soup] =>' in text)

		with io.StringIO() as out:
			self.smodel.print_graph(out, sformat='{%t |= hasCake}', eformat='%l %l')
			text = out.getvalue()
			self.assertTrue(text.startswith('digraph {'))
			self.assertTrue('[label="true"]' in text)
			self.assertTrue('[label="cake cake"]' in text)

	def test_check(self):
		"""Standard model checking"""

		f = self.m.parseTerm('<> hasCake')

		self.assertFalse(self.model.check(f)[0])
		self.assertFalse(self.model.check('<> hasCake')[0])
		self.assertTrue(self.model.check('E <> hasCake')[0])
		self.assertTrue(self.model.check('E <> hasCake', backends=('builtin', ))[0])

		bmodel = api.MaudeModel(self.t, strategy=self.b)

		self.assertTrue(self.smodel.check('A O E <> hasCake')[0])
		self.assertFalse(self.smodel.check('A O E <> hasCake', merge_states='no')[0])

	def test_pcheck(self):
		"""Probabilistic model checking"""

		self.assertEqual(self.model.pcheck('<> hasCake')[0].value, 0.25)
		self.assertFalse(self.model.pcheck('P >= 0.5 hasCake')[0].value)

	def test_scheck(self):
		"""Statistical model checking"""

		# Literal QuaTEx file
		quatex_file = '''HasCake(n) =
			if (n == 0) then s.rval("M:Machine |= hasCake")
			            else # HasCake(n - 1) fi ;
			eval E[HasCake(2)] ;
		'''

		with io.StringIO(quatex_file) as quatex:
			result = self.model.scheck(quatex, assign='uaction(put1=3)')

			self.assertTrue(result['nsims'] >= 30)
			self.assertEqual(result['queries'][0]['mean'], 0.0)


if __name__ == '__main__':
	unittest.main()
