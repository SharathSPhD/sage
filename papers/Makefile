# ---------------------------------------------------------------------------
# Build the three papers to PDF.
#
#   make -C papers papers      # figures + all three PDFs
#   make -C papers p2          # just the flagship
#   make -C papers lint        # chktex over every source file
#   make -C papers clean       # remove LaTeX build litter (keeps the PDFs)
#
# TOOLCHAIN REQUIRED (see README.md in this directory):
#   pdflatex + latexmk from TeX Live, with the latex-recommended, latex-extra,
#   fonts-recommended and mathscience collections. chktex is optional and only
#   needed by `make lint`.
#
#   No root, TeX Live user install (this is what the DGX Spark runs):
#     tlmgr install collection-latexrecommended collection-latexextra \
#                   collection-fontsrecommended collection-mathscience \
#                   collection-bibtexextra latexmk chktex
#   Debian/Ubuntu with root:
#     sudo apt-get install -y --no-install-recommends texlive-latex-recommended \
#       texlive-latex-extra texlive-fonts-recommended texlive-science \
#       latexmk chktex
#
# Every paper carries its bibliography inline (thebibliography), so no BibTeX
# or biber pass is needed; latexmk still runs pdflatex twice for references.
# ---------------------------------------------------------------------------

LATEXMK      ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
PYRUN        ?= uv run python

.PHONY: papers p1 p2 p3 figures lint clean help

help:
	@echo "targets: papers | p1 | p2 | p3 | figures | lint | clean"

papers: p1 p2 p3

figures:
	$(PYRUN) p1_instruments/make_figures.py
	$(PYRUN) p2_plane/make_figures.py
	$(PYRUN) p3_noneq/make_figures.py

p1:
	cd p1_instruments && $(LATEXMK) $(LATEXMKFLAGS) p1_main.tex

p2:
	cd p2_plane && $(LATEXMK) $(LATEXMKFLAGS) p2_main.tex

p3:
	cd p3_noneq && $(LATEXMK) $(LATEXMKFLAGS) p3_main.tex

lint:
	@for f in p1_instruments/*.tex p1_instruments/sections/*.tex \
	          p2_plane/*.tex p2_plane/sections/*.tex \
	          p3_noneq/*.tex p3_noneq/sections/*.tex; do \
	    chktex -q -n8 -n24 "$$f" || true; \
	done

clean:
	cd p1_instruments && $(LATEXMK) -c p1_main.tex || true
	cd p2_plane      && $(LATEXMK) -c p2_main.tex || true
	cd p3_noneq      && $(LATEXMK) -c p3_main.tex || true
