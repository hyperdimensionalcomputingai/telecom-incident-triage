# Hyperdimensional Computing for Telecom Incident Triage

This repo contains a synthetic telecom incident scenario that an analyst may be looking to understand using [Hyperdimensional Computing](https://hyperdimensionalcomputing.ai/) (HDC). Imagine that a commuter's call drops. Which earlier incidents resemble it, what network facts support the comparison, and how can reviewed incidents improve future triage?

The HDC experiments shown here use public Toronto street geometry with simulated subscribers, phones, network connections and measurements. Binding, bundling and permutation, the three fundamental operators of HDC, are used to build a composable, high-dimensional representation for retrieval and incremental class memory updates. In HDC, learning is akin to updating a memory, and its results in these experiments are compared against well-known techniques from classical ML: regularized logistic regression (LR) and a small neural network (an MLP).

The experiments examine four questions:

- **Representation:** Do roles, event order and network connectivity change the meaning of an encoded incident?
- **Retrieval and evidence:** Can it find comparable earlier incidents and return the source records supporting the comparison?
- **Learning:** How useful does class memory become as reviewed examples arrive, and when do updates help or hurt?
- **Resources:** How long do prediction and learning from new reviews take, and how much storage does each representation need?

The Python files live directly in [src/](src/).

## Reproduce

Requires **Python 3.13** and **uv**. From the repository root:

```sh
uv sync
uv run src/prepare.py
uv run src/run.py
uv run src/report.py
```

The scripts share two paths in [src/settings.py](src/settings.py): `CONFIG_FILE` selects `configs/tutorial.json`, and `RUN_DIR` selects `runs/my-run`. Open `runs/my-run/reports/summary.md` for your results. Choose a new output directory for each run.

The full study crosses five data seeds with three encoder seeds, with 2,400 episodes per dataset. For a quick pipeline check, change `CONFIG_FILE` to `configs/smoke.json`; its small dataset is not intended for performance claims. Preparation uses the bundled street snapshot.

Run the acceptance checks with `uv run pytest -q`.

## Edit the report narrative

The report's editable prose lives in [docs/report-templates/](docs/report-templates/). The main template is [summary.md](docs/report-templates/summary.md), with separate Markdown sections for the encoder, learning updates and potential geographic enhancements.

Named placeholders such as `{{ retrieval_rows }}` mark where Python inserts calculated values or table rows. Figure links stay in the Markdown; the reporting script generates the images at those paths. These templates use simple named replacement, with no Jinja dependency or expressions to evaluate. A missing placeholder value stops report generation with a clear error.

Edit the templates, then run `uv run src/report.py` for a completed run to rebuild its report. Keep prose edits in the templates: the generated `summary.md` is overwritten when reporting runs again. The report's provenance records hashes of the Python generator and Markdown templates.

## Results

Read the [illustrated results summary](runs/reports/summary.md) for the encoder equations, graph schema, experiments and findings. The latest [aggregate results](runs/summary.json) and [detailed metrics](runs/metrics.json) are committed with their configuration and provenance; datasets, vectors and individual prediction traces are regenerated locally.

- Roles, order and connectivity matter; retrieved comparisons resolve to retained source evidence.
- HDC retrieval achieves **96.3% precision@5**, with source records supporting each returned comparison.
- With just one review per class, HDC scores **7–9 points higher macro F1** than LR and the MLP (82.3% versus 74.9% and 73.4%). From five reviews per class, all three are on par, within about 2 points.
- HDC learns from a single new review in about 0.13 ms, versus about 3.7 ms for an LR refit and 40 ms for an MLP refit. One refit can absorb a whole batch, though: at 40 reviews, LR is faster. All three predict in about 0.1 ms, and HDC uses more storage.

These findings concern controlled synthetic patterns, and do not represent the operational carrier performance of any real provider. The run times reported in the results depend on the machine being used, so use them as guidance only.

Street geometry: [Toronto Centreline](https://ckan0.cf.opendata.inter.prod-toronto.ca/en/dataset/toronto-centreline-tcl). The dataset contains information licensed under the [Open Government Licence – Toronto](https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-licence/).
