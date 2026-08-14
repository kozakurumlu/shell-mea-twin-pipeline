#!/bin/sh
# Compile the NLDL paper: pdflatex -> biber -> pdflatex x2.
# Needs a TeX distribution with biber (TeX Live / TinyTeX / MacTeX).
# To use a TeX that is not on PATH, create an untracked local_tex.sh here
# containing:  PATH="/path/to/tex/bin:$PATH"; export PATH
cd "$(dirname "$0")" || exit 1
[ -f local_tex.sh ] && . ./local_tex.sh
pdflatex -interaction=nonstopmode main.tex && \
biber main && \
pdflatex -interaction=nonstopmode main.tex && \
pdflatex -interaction=nonstopmode main.tex && \
echo "Done: main.pdf"
