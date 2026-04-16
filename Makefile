#
# Makefile for the umaudemc tool
#

# Python command to be included in the shebang
PYTHON ?= /usr/bin/env python3

# Bundle the all the Python and data file into single executable zip file
# (based on Stack Overflow's question 17486578)

RESOURCES = umaudemc/data/*.maude umaudemc/data/*.js umaudemc/data/*.css umaudemc/data/*.htm
CODE      = umaudemc/*.py umaudemc/*/*.py

dist/umaudemc: dist $(RESOURCES) $(CODE)
	# Create temporary directory and copy the package into it
	mkdir -p zip
	cp --parents $(CODE) $(RESOURCES) zip/
	# Create a __main__ file for the package that invokes the umaudemc one
	echo -e 'import sys\nfrom umaudemc.__main__ import main\nsys.exit(main())' > zip/__main__.py
	# Set timestamps to current time
	touch -ma zip/* zip/*/*
	# Compress that directory into a zipapp file
	python -m zipapp -c -p "$(PYTHON)" zip -o $@
	rm -rf zip
	chmod a+x $@

wheel:
	pip wheel --no-deps -w dist .
	$(RM) -r build umaudemc.egg-info

dist:
	mkdir -p dist
