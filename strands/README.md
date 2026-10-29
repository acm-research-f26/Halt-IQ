# strands/

Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`) as a HaltIQ critic, with a
MiniLM / bge reranker choosing the evidence paragraphs. Copied from `strands-reranker/` on the
`vedanshi` branch (commit d0688ae).

- `Vedanshi_StrandsCritic.ipynb`: Colab notebook (GPU) that serves the decider and scores the 220 dev drafts.
- `metrics.py`: recomputes `results/` from the saved CSVs, no model calls. It imports
  `critic-bench/report.py`, which is not on this branch yet, so it does not run here as-is.
- `results/`: per-draft scores (`haltiq_strands_experiments*.csv`), metrics tables, AUROC chart.
- `WRITEUP_DRAFT.md`, `WRITEUP_TEMPLATE.md`: write-up and the template `metrics.py` fills.

Not yet included: the Oct 23 replay of stopping rules over Sri's 150 trajectories and its
per-draft Strands+bge scores. Those files were not found in the repo or on this machine.
