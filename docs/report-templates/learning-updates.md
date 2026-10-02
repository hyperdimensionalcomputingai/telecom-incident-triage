### Learning from newly reviewed incidents

**Setup.** Every method starts from the same {{ initial_reviews }} reviewed incidents ({{ max_reviews_per_class }} per class). A batch of {{ batch_sizes }} later reviews then arrives, using the same review IDs for every method. Each method encodes the new incidents, learns from their labels, and predicts them with the updated memory or model. The timer covers all three steps.

The methods learn differently. HDC adds each new hypervector to its class memory and never revisits earlier reviews. LR and the MLP have no incremental update in this setup, so they refit once on every retained review: {{ refit_sizes }} examples, with the frozen settings. Each of the {{ refit_repeats }} measured repetitions (after {{ refit_warmup }} warm-ups) starts from the same state.{{ batch_coverage }}

![Total time to learn from a batch of newly reviewed incidents](figures/learning-updates.png)

Total time to encode, learn from and predict the whole batch:

| New reviews | HDC addition, ms | LR refit, ms | MLP refit, ms |
|---:|---:|---:|---:|
{{ learning_update_rows }}

Where that time goes when a single review arrives:

| Method | Encode, ms | Learn, ms | Predict, ms | Total, ms |
|---|---:|---:|---:|---:|
{{ learning_update_stage_rows }}

{{ learning_update_finding }}

A batch total divided by its size gives a throughput figure, but no review experiences that time: with a refit, every review in the batch waits until the whole refit finishes, plus however long the batch took to collect. We therefore report batch totals only.

Stage medians need not sum exactly to the median total. Timed HDC updates are checked, outside the timer, against a complete reconstruction from the initial and new hypervectors. There are {{ refit_warning_count }} convergence-warning refits among the measured repetitions; their diagnostics and all raw timings are retained under `resources.learning_updates` in every pair's `results.json`. Because the starting history is fixed at {{ initial_reviews }} reviews, these results do not show how refit cost scales with much larger histories.
