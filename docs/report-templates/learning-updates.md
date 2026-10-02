### Next evaluation: per-sample incremental learning versus batch retraining

**Question.** Starting from the same reviewed history, how much compute and latency does each method need to incorporate newly reviewed incidents? This requested follow-up is now a completed experiment.

**Setup.** Each measurement starts with {{ initial_reviews }} reviewed incidents ({{ max_reviews_per_class }} per class), using the same review IDs across methods. Batches contain {{ batch_sizes }} chronologically later memory-partition reviews. For each batch, every method encodes the new incidents, incorporates their labels, then predicts that batch with the updated memory or model. Final-test and validation reviews do not participate.

HDC normalizes each new hypervector and adds it to the labelled class accumulator; it does not revisit earlier incidents or optimize an encoder. LR and the MLP reuse the retained earlier feature vectors and perform **one full L-BFGS refit per arriving batch**, on the initial reviews plus that batch, with the already frozen regularization and model settings. Twenty new reviews therefore mean one refit on {{ reviews_after_twenty }} reviews, rather than twenty separate refits; forty mean one refit on {{ reviews_after_forty }}.

Each repetition resets to the same starting state. Initial fitting and HDC state restoration occur outside the timer. All methods use one Torch/BLAS CPU thread, {{ refit_warmup }} warm-up repetitions and {{ refit_repeats }} measured repetitions per batch and seed pair, with a warm encoder cache. The measurements include feature/vector encoding, learning, and prediction; joins, geographic eligibility, audit persistence and database work remain separate. {{ batch_coverage }}

For $B$ new reviews and $N$ previously reviewed incidents, the measured operations are:

$$
T_{\mathrm{HDC}}(B)=T_{\mathrm{encode}}(B)+T_{\mathrm{normalize+add}}(B)+T_{\mathrm{predict}}(B),
$$

$$
T_{\mathrm{LR/MLP}}(N,B)=T_{\mathrm{encode}}(B)+T_{\mathrm{refit}}(N+B)+T_{\mathrm{predict}}(B),
\qquad t_{\mathrm{per\ new\ review}}=\frac{T(N,B)}{B}.
$$

The per-review number for LR/MLP is **amortized batch cost**. It is not the time to update immediately when each sample arrives; collecting a batch introduces a waiting time that this compute benchmark does not measure. For HDC, the same additions can be applied one review at a time without waiting for a batch. The delayed-feedback experiment demonstrates that separate behaviour.

**Result.** Total wall latency and amortized cost per new review, reported as the median of seed-pair medians. The p95 column is the median of seed-pair p95 measurements; it is descriptive timing variation, not an independent-world confidence interval. Process CPU time measures CPU work during the complete operation alongside elapsed wall time.

![Complete batch latency and amortized cost per newly reviewed incident](figures/learning-updates.png)

| New reviews | Learning method | Reviews after update | Total wall, ms | Wall p95, ms | Wall per new review, ms | CPU per new review, ms |
|---:|---|---:|---:|---:|---:|---:|
{{ learning_update_rows }}

The corresponding stage costs per new review are:

| New reviews | Learning method | Encoding, ms | Addition or refit, ms | Prediction, ms |
|---:|---|---:|---:|---:|
{{ learning_update_stage_rows }}

{{ learning_update_finding }}

Stage medians need not sum exactly to the median complete-operation latency. Raw repeated measurements and fit diagnostics are retained under `resources.learning_updates` in every pair's `results.json`. Timed HDC updates must exactly match a complete reconstruction from the initial and newly reviewed hypervectors; those checks occur outside the timer. There are {{ refit_warning_count }} convergence-warning refits among the measured batch repetitions; diagnostics are retained rather than silently discarded.

**Interpretation.** This compares additive HDC class-memory updates with these implementations' full batch retraining strategy, including the cost of encoding new incidents. It does not benchmark incremental SGD, warm starts, cached encoder-free updates, energy or a production pipeline. HDC's update work depends on the incoming hypervectors and fixed class memory, while a batch refit consumes the growing retained review set. Because the initial history is fixed at {{ initial_reviews }} reviews, the experiment measures the requested batch costs; it does not establish a scaling law over arbitrarily large training histories.
