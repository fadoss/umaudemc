import setuptools

long_description = '''# Unified Maude model-checking tool

Uniform interface for model checking LTL, CTL, CTL*, and μ-calculus properties
on standard and [strategy](https://maude.ucm.es/strategies)-controlled
[Maude](https://maude.cs.illinois.edu) specifications using built-in and
external backends. Models can also be extended with quantitative information
and be applied probabilistic and statistical model-checking techniques.

This tool can be used from the command line, from a graphical user interface,
and as a Python library. See the
[repository](https://github.com/fadoss/umaudemc) for additional information,
documentation, and examples.
'''

setuptools.setup(
	name                          = 'umaudemc',
	version                       = '0.12',
	author                        = 'ningit',
	author_email                  = 'ningit@users.noreply.github.com',
	description                   = 'Unified Maude model-checking utility',
	long_description              = long_description,
	long_description_content_type = 'text/markdown',
	url                           = 'https://github.com/fadoss/umaudemc',
	project_urls                  = {
		'Bug Tracker'   : 'https://github.com/fadoss/umaudemc/issues',
		'Documentation' : 'https://github.com/fadoss/umaudemc',
		'Source Code'   : 'https://github.com/fadoss/umaudemc'
	},
	license                       = 'GPLv3',
	packages                      = setuptools.find_packages(),
	package_data                  = {'': ['data/*']},
	include_package_data          = True,
	entry_points                  = {
		'console_scripts': [
			'umaudemc=umaudemc.__main__:main'
		]
	},
	classifiers                   = [
		'Programming Language :: Python :: 3',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Topic :: Scientific/Engineering',
		'Operating System :: OS Independent',
	],
	python_requires               = '>=3.7',
	install_requires              = [
		'maude >= 1.0',
	],
	extras_require                = {
		'CTL*'	: ['pyModelChecking >= 1.3.3'],
		'YAML'	: ['pyaml'],
		'Plot'  : ['matplotlib'],
		'SMC'   : ['scipy'],
	}
)
