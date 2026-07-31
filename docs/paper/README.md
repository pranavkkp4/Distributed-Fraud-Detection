# Evidence-first paper source

`research_paper.tex` is the A4 two-column source for the project research
paper. The checked-in 10-page PDF combines the implementation design with the
bounded CPU shared-memory and local CUDA component measurements retained under
`docs/profiling/results/`. Historical V0--V5 targets remain explicitly separate
from observed values.

Build from the repository root:

```powershell
python scripts/generate_research_figures.py
xelatex -interaction=nonstopmode -output-directory=docs/paper docs/paper/research_paper.tex
xelatex -interaction=nonstopmode -output-directory=docs/paper docs/paper/research_paper.tex
```

The generator reads only `docs/profiling/results/*.json` and `*.csv`. A JSON
manifest must contain populated `results` fields to appear as a measured run;
otherwise the rendered paper says **not measured**. Store raw profiler exports,
load manifests, quantile CSVs, and hardware/software provenance alongside the
manifest before presenting numerical results.

The current PDF reports a single seeded Linux HTTP/System V integration run and
five local RTX 3050 component runs. It does not claim a full-model production
SLO. Re-run the generator before every build so added or removed result
manifests cannot leave stale tables or figures in the paper.
