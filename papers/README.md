# Papers

Three papers, each self-contained in its own directory with a `sections/` split,
a `figures/` directory of committed PDFs, and a `make_figures.py` that
regenerates those figures from committed artifacts.

| Directory | Paper | Target venue |
|---|---|---|
| `p1_instruments/` | `strataq` — the software paper | JOSS / SoftwareX |
| `p2_plane/` | The Irreversibility Plane (flagship) | MDPI *Entropy* / *Symmetry* |
| `p3_noneq/` | Non-equilibrium thermodynamics of quantal response | working draft |

## Building

```bash
make -C papers papers      # regenerate figures, then build all three PDFs
make -C papers p2          # just the flagship
make -C papers lint        # chktex over every source file
make -C papers clean       # remove .aux/.log/.fls litter, keep the PDFs
```

## Toolchain requirement

Building needs a **TeX Live installation with `pdflatex` and `latexmk`**, plus the
packages used by the preambles: `geometry`, `amsmath`/`amssymb`/`amsthm`,
`booktabs`, `graphicx`, `xcolor`, `float`, `caption`, `hyperref`, `natbib`.
That is the `latex-recommended`, `latex-extra`, `fonts-recommended` and
`mathscience` collections. `chktex` is only needed for `make lint`.

Install without root (TeX Live / TinyTeX user install — this is what the DGX
Spark runs, at `~/.TinyTeX`, with `~/.local/bin` on `PATH`):

```bash
tlmgr install collection-latexrecommended collection-latexextra \
              collection-fontsrecommended collection-mathscience \
              collection-bibtexextra latexmk chktex
```

Install with root on Debian/Ubuntu:

```bash
sudo apt-get install -y --no-install-recommends \
  texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
  texlive-science latexmk chktex
```

Each paper carries its bibliography inline as a `thebibliography` environment,
so no BibTeX or biber run is required; `latexmk` still needs two `pdflatex`
passes to resolve `\ref` and `\cite`.

## Figures

`make figures` runs each paper's `make_figures.py` through `uv run python`, so
the repository's environment must be synced (`uv sync --all-packages`) first.
`p2_plane/figures/figure_sources.json` records which artifact each figure was
drawn from.
