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
	cp -r umaudemc zip
	# Create a __main__ file for the package that invokes the umaudemc one
	echo -e 'import sys\nfrom umaudemc.__main__ import main\nsys.exit(main())' > zip/__main__.py
	touch -ma zip/* zip/*/*
	# Compress that directory into a zip file
	cd zip ; zip -q ../umaudemc.zip $(RESOURCES) $(CODE) __main__.py
	rm -rf zip
	# Put the shebang and then the zip file into the executable bundle
	echo '#!$(PYTHON)' > $@
	cat umaudemc.zip >> $@
	rm umaudemc.zip
	chmod a+x $@

wheel:
	pip wheel --no-deps -w dist .

dist:
	mkdir -p dist
