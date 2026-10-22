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

Requires **Python 3.13** and **uv**. [lance-graph](https://github.com/lancedb/lance-graph) only publishes Linux x86_64 wheels, so elsewhere `uv sync` builds it from source, which needs a Rust toolchain and `protoc`. From the repository root:

```sh
uv sync --locked
uv run src/prepare.py
uv run src/run.py
uv run src/report.py
```

The scripts share two paths in [src/settings.py](src/settings.py): `CONFIG_FILE` selects `configs/tutorial.json`, and `RUN_DIR` selects `runs/my-run`. Open `runs/my-run/reports/summary.md` for your results. Choose a new output directory for each run.

The full study crosses five data seeds with three encoder seeds, with 2,400 episodes per dataset. For a quick pipeline check, change `CONFIG_FILE` to `configs/smoke.json`; its small dataset is not intended for performance claims. Preparation uses the bundled street snapshot.

Run the acceptance checks with `uv run pytest -q`.

## Follow the evidence trail

HDC similarity search returns a comparable earlier episode. To see the facts behind the match, [src/storage.py](src/storage.py) uses [lance-graph](https://github.com/lancedb/lance-graph) to read the stored Lance source tables as a property graph and runs a Cypher query from the retrieved episode. Each relationship reuses a table that already holds both endpoint IDs (for example, `SERVED_BY` is the `observations` table's `observation_id` → `cell_id`), so the graph needs no hand-written joins and no second copy of the data.

```cypher
MATCH (e:Episode)-[:HAS_OBSERVATION]->(o:Observation)-[:SERVED_BY]->(c:Cell)-[b:BACKHAULED_BY]->(l:Link),
      (o)-[:FROM_PHONE]->(ph:Phone),
      (c)-[:CELL_STATE]->(cs:CellStatus),
      (l)-[:LINK_STATE]->(ls:LinkStatus)
WHERE e.episode_id = $episode
  AND o.timestamp_s >= b.valid_from_s AND o.timestamp_s < b.valid_to_s
  AND cs.timestamp_s = o.timestamp_s AND ls.timestamp_s = o.timestamp_s
RETURN o.ordinal, o.radio_dbm, ph.handset, cs.cell_load_pct, ls.link_loss_pct, ls.link_latency_ms
ORDER BY o.ordinal
```

The path follows the phone's signal to its serving cell's load, then to the backhaul link that was connected at that time and its loss and latency. A second query continues from that link to the other phones sharing it. From Python:

```python
hits = vectors.search(raw_query, eligible, decision_s)  # HDC similarity search
trail = sources.evidence_trail(hits[0]["episode_id"])  # Cypher over the same Lance tables
trail["path"]   # phone signal, cell load, backhaul loss and latency at each of three times
trail["peers"]  # other phones on that backhaul link at the same times
```

During a run, the trail for every top-ranked match must reach exactly the source records its hypervector encoded; any disagreement stops the run.

## Edit the report narrative

The report's editable prose lives in [docs/report-templates/](docs/report-templates/). The main template is [summary.md](docs/report-templates/summary.md), with separate Markdown sections for the encoder, learning updates and potential geographic enhancements.

Named placeholders such as `{{ retrieval_rows }}` mark where Python inserts calculated values or table rows. Figure links stay in the Markdown; the reporting script generates the images at those paths. These templates use simple named replacement, with no Jinja dependency or expressions to evaluate. A missing placeholder value stops report generation with a clear error.

Edit the templates, then run `uv run src/report.py` for a completed run to rebuild its report. Keep prose edits in the templates: the generated `summary.md` is overwritten when reporting runs again. The report's provenance records hashes of the Python generator and Markdown templates.

## Results

Read the [illustrated results summary](runs/reports/summary.md) for the encoder equations, graph schema, experiments and findings. The latest [aggregate results](runs/summary.json) and [detailed metrics](runs/metrics.json) are committed with their configuration and provenance; datasets, vectors and individual prediction traces are regenerated locally.

**Where HDC adds value**

- **Learns from very few reviews.** With one review per class, HDC scores 7–9 points higher macro F1 than LR and the MLP (82.3% versus 74.9% and 73.4%).
- **Learning from a new review costs almost nothing, and the cost does not grow.** About 0.13 ms per review, versus about 3.7 ms to retrain LR and 40 ms to retrain the MLP on every update; retraining time also grows with the training set.
- **Finds comparable incidents and shows why.** 96.3% precision@5 (random: 25%), and every similarity score breaks down into contributions traceable to source records.

**Tradeoffs:** hypervectors need far more storage than raw measurements, and from five reviews per class all three methods are on par. See the summary's [key findings](runs/reports/summary.md#key-findings).

These findings concern controlled synthetic patterns, and do not represent the operational carrier performance of any real provider. The run times reported in the results depend on the machine being used, so use them as guidance only.

Street geometry: [Toronto Centreline](https://ckan0.cf.opendata.inter.prod-toronto.ca/en/dataset/toronto-centreline-tcl). The dataset contains information licensed under the [Open Government Licence – Toronto](https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-licence/).
