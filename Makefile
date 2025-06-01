# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = docs
BUILDDIR     = docs/_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for development
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)/*

livehtml:
	@echo "Starting live reload server..."
	@sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)

install-deps:
	@echo "Installing documentation dependencies..."
	@pip install -r docs/requirements.txt
