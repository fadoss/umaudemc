#
# Statistical methods for calculating confidence intervals
#

import math

from .common import usermsgs, maude

# Global store of singleton methods
singletons = {}

#
# Each statistical method is a class with two methods:
#
# - check checks whether the experiment has converged and sets
#   query.converged accordingly
#
# - init initialize the method and returns the calculated number
#   of exections or None otherwise
#

class ChowRobbinsStudentMethod:
	"""Chow-Robbins method using the Student-t distribution"""

	__slots__ = ('stdtrit',)

	NAME = 'Chow-Robbins'

	def __init__(self):
		from scipy.special import stdtrit
		self.stdtrit = stdtrit

	def check(self, query, alpha, min_sim):
		# Radius of the confidence interval calculated with the Student-t distribution
		query.h = query.s * (self.stdtrit(query.n - 1, 1 - alpha / 2) / math.sqrt(query.n))

		if query.h <= query.delta and query.n >= min_sim:
			query.converged = True

	def init(self, delta, alpha, min_sim):
		return None


class ChowRobbinsNormalMethod:
	"""Chow-Robbins method usings the normal distribution"""

	__slots__ = ('inv_cdf',)
	NAME = 'Chow-Robbins (with normal distribution)'

	def __init__(self):
		from statistics import NormalDist
		self.inv_cdf = NormalDist().inv_cdf

	def check(self, query, alpha, min_sim):
		# The same as ChowRobbinsStudentMethod, but with the normal distribution
		query.h = query.s * (self.inv_cdf(1 - alpha / 2) / math.sqrt(query.n))

		if query.h <= query.delta and query.n >= min_sim:
			query.converged = True

	def init(self, delta, alpha, min_sim):
		return None


class SeqClopperPearsonMethod:
	"""Sequential Clopper-Pearson method"""

	__slots__ = ('ppf',)

	NAME = 'sequential Clopper-Pearson'

	def __init__(self):
		from scipy.stats import beta
		self.ppf = beta.ppf

	def check(self, query, alpha, min_sim):
		# Lower and upper bounds of the confidence interval
		bmin = self.ppf(alpha / 2, query.sum, query.n - query.sum + 1) if query.sum > 0 else 0.0
		bmax = self.ppf(1 - alpha / 2, query.sum + 1, query.n - query.sum) if query.n > query.sum else 1.0

		query.h = (bmax - bmin) / 2

		if query.h <= query.delta and query.n >= min_sim:
			query.converged = True

	def init(self, delta, alpha, min_sim):
		return None


class ClopperPearsonMethod:
	"""Two-phase Clopper-Pearson method"""

	# From Algorithm 1 of "Clopper-Pearson algorithms for efficient statistical
	# model checking estimation" by Hao Bu and Meng Sun (DOI: 10.1109/TSE.2024.3392720)

	__slots__ = ('ppf', 'check')

	NAME = 'Clopper-Pearson'

	def __init__(self):
		from scipy.stats import beta
		self.ppf = beta.ppf
		self.check = self._phase1

	def _phase1(self, query, alpha, min_sim):
		# Lower and upper bounds of the confidence interval
		bmin = self.ppf(alpha / 2, query.sum, query.n - query.sum + 1) if query.sum > 0 else 0.0
		bmax = self.ppf(1 - alpha / 2, query.sum + 1, query.n - query.sum) if query.n > query.sum else 1.0

		query.h = (bmax - bmin) / 2

		if query.h <= query.delta and query.n >= min_sim:
			query.converged = True

	def init(self, delta, alpha, min_sim):
		return None


class ChenMethod:
	"""Chen method"""

	# From Algorithm 1 of "Properties of a new adaptive sampling method with
	# applications to scalable learning" by Jianhua Chen (DOI: 10.3233/WEB-150322)

	__slots__ = ('check', 'nsims', 'delta')

	NAME = 'Chen'

	def __init__(self):
		self.check = self._phase1
		self.nsims = 0

	def _phase1(self, query, alpha, min_sim):
		"""Phase 1 (Algorithm 1)"""

		w = 0.5 - abs(0.5 - query.mu)
		needed = math.log(alpha / 2) / self.u(w, w + query.delta)

		# Done with phase 1, go to phase 2
		if query.n >= max(needed, min_sim):
			self.check = self._phase2

			# Calculation as in Algorithm M1
			delta = math.sqrt(math.log(2 / alpha) / (2 * query.n))
			p = query.mu + math.copysign(1.0, .5 - query.mu) * (delta - self.delta)
			w = 0.5 - abs(0.5 - p)

			self.nsims = max(math.ceil(math.log(alpha / 2) / self.u(w, w + query.delta)), min_sim)

	def _phase2(self, query, alpha, min_sim):
		"""Phase 2 of Algorithm M1"""

		if query.n >= self.nsims:
			query.h = self.delta
			query.converged = True

	def init(self, delta, alpha, min_sim):
		# Save delta for the second phase calculation
		self.delta = delta

		return None

	@staticmethod
	def u(z, theta):
		if z == 0:
			return math.log(1 - theta)
		if z == 1:
			return math.log(theta)
		if not (0 < theta < 1):
			return -math.inf

		return z * math.log(theta / z) + (1 - z) * math.log((1 - theta) / (1 - z))


class HoeffdingMethod:
	"""Hoeffding inequality method"""

	__slots__ = ('diameter', 'factor', 'nsims')

	NAME = "Hoeffding's inequality"

	def __init__(self, diameter):
		self.diameter = diameter  # diameter of the image
		self.factor = 1.0  # factor for recomputing the interval size
		self.nsims = 0  # number of required simulations

	def check(self, query, alpha, min_sim):
		query.h = self.factor / math.sqrt(query.n)

		if query.n >= self.nsims:
			query.converged = True

	def init(self, delta, alpha, min_sim):
		# delta is given
		if delta is not None:
			# Calculate the number of simulations according to Hoeffding's inequality
			nsims = math.ceil(self.diameter ** 2 * math.log(2 / alpha) / (2 * delta ** 2))
			self.nsims = max(min_sim, nsims)

		else:
			self.nsims = min_sim

		# Calculate the factor to avoid recomputing it in check
		self.factor = self.diameter * math.sqrt(math.log(2 / alpha) / 2)

		return self.nsims


def print_error_smethod(msg):
	"""Print error from a malformed smethod"""

	return usermsgs.print_error(f'{msg[0].upper()}{msg[1:]}')


def get_statistical_method(method, qinfo=None):
	"""Get the function that calculates the radius of a confidence interval"""

	radius_func = None

	# Get the method object from the singleton store (if present)
	if method_obj := singletons.get(method):
		return method_obj

	# Explanation for errors due to missing SciPy
	message = 'using the Chow-Robbins method with the normal distribution'

	# Error function
	error_fn = print_error_smethod if qinfo is None else lambda msg: usermsgs.print_error_loc(qinfo.filename, qinfo.line, qinfo.column, msg)

	try:
		match method:
			# Sequential Clopper-Pearson (for probabilities)
			case 'seq-clopper-pearson' | 'scp':
				method_obj = SeqClopperPearsonMethod()

				singletons['seq-clopper-pearson'] = method_obj
				singletons['scp'] = method_obj

			# Two-phases Clopper-Pearson (for probabilities)
			case 'clopper-pearson' | 'cp':
				method_obj = ClopperPearsonMethod()

			# Okamoto bound method (for probababilities)
			case 'okamoto':
				method_obj = HoeffdingMethod(1.0)

			# Chen method (Algorithm 1, for probabilities)
			case 'chen':
				method_obj = ChenMethod()

				singletons['chen'] = method_obj

			# Chow-Robbins method
			case 'chow-robbins' | 'cr':
				message = 'using the normal distribution instead of Student\'s t-distribution'
				method_obj = ChowRobbinsStudentMethod()

				singletons['chow-robbins'] = method_obj
				singletons['cr'] = method_obj

			# Hoeffding method or unknown method
			case _:
				# Unknown statistical method, this is a hard error
				if not method.startswith('hoeffding'):
					error_fn(f'unknown statistical method {method}')
					return None

				# Check hoeffding argument format
				if not method.startswith('hoeffding(') and method[-1] == ')':
					error_fn('bad method specification in {method}: hoeffding(length) is expected')
					return None

				try:
					length = float(method[10:-1])

				except ValueError:
					error_fn('cannot parse Hoeffding argument {method[10:-1]} as a number')
					return None

				method_obj = HoeffdingMethod(length)


	# SciPy is required for obtaining the quantiles of the Student's
	# t-distribution and the beta, although it is a huge dependency including NumPy
	except ImportError:
		pass

	if method_obj is None:
		usermsgs.print_warning(f'Warning: {message} for calculating confidence intervals, '
		                       'since SciPy is not available.')

		method_obj = ChowRobbinsNormalMethod()
		singletons[method] = method_obj
		singletons['chow-robbins'] = method_obj
		singletons['cr'] = method_obj

	return method_obj
