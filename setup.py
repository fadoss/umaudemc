import setuptools

setuptools.setup(
	name				= 'umaudemc',
	version				= '0.1',
	author				= 'ningit',
	author_email			= 'ningit@users.noreply.github.com',
	description			= 'Unified Maude model-checking utility',
	long_description 		= 'Unified tool to model check Maude models',
	long_description_content_type	= 'text/markdown',
	url				= 'http://maude.ucm.es/strategies',
	project_urls                    = {
		'Bug Tracker': 'https://github.com/fadoss/umaudemc/issues',
		'Source Code': 'https://github.com/fadoss/umaudemc'
	},
	license				= 'GPLv3',
	packages			= setuptools.find_packages(),
	package_data			= {'': ['data/*']},
	include_package_data		= True,
	classifiers			= [
		'Programming Language :: Python :: 3',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
	        'Topic :: Scientific/Engineering',
		'Operating System :: OS Independent',
	],
	python_requires			= '>=3.7',
	install_requires		= [
		'maude >= 0.4',
	],
	extras_requires			= {
		'CTL*'   : ['pyModelChecking >= 1.3.3'],
		'YAML'   : ['pyaml']
	}
)
