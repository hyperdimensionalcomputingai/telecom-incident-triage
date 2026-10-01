"""Matched reviewed batches: additive memory versus full retained-review refits."""

from time import perf_counter_ns, process_time_ns

import torch

from classifiers import METHODS, fit_classifier
from encoding import model_input_features
from evaluation import latency_summary
from learning import Prototype, budget_indices


def learning_update_cost(encoder, episodes, raws, features, labels, memory, config, settings, seed):
    if any(episodes[i]["split"] != "memory" for i in memory):
        raise ValueError("Update-cost measurements require memory-partition reviews only")
    budget = max(config.budgets)
    selected = budget_indices(memory, labels, budget, episodes)
    cutoff = max((episodes[i]["decision_s"], episodes[i]["episode_id"]) for i in selected)
    remaining = sorted(
        (
            i
            for i in memory
            if i not in selected and (episodes[i]["decision_s"], episodes[i]["episode_id"]) > cutoff
        ),
        key=lambda i: (episodes[i]["decision_s"], episodes[i]["episode_id"]),
    )
    base = Prototype(encoder.dimension)
    for index in selected:
        base.update(raws[index], int(labels[index]))
    rows, skipped = [], []
    for batch_size in config.update_batches:
        if len(remaining) < batch_size:
            skipped.append(
                {
                    "batch_size": batch_size,
                    "reason": "Not enough later memory reviews",
                    "available": len(remaining),
                }
            )
            continue
        extra = remaining[:batch_size]
        expected = Prototype(encoder.dimension)
        expected.values.copy_(base.values)
        expected.counts.copy_(base.counts)
        for index in extra:
            expected.update(raws[index], int(labels[index]))
        for method in ("hdc", *METHODS):
            samples = {
                key: []
                for key in (
                    "encoding_ms",
                    "learning_ms",
                    "prediction_ms",
                    "complete_wall_ms",
                    "complete_cpu_ms",
                )
            }
            fit_diagnostics = []
            for repeat in range(config.refit_warmup + config.refit_repeats):
                # Reset outside the timer: every repetition starts with the same 80-review memory.
                if method == "hdc":
                    model = Prototype(encoder.dimension)
                    model.values.copy_(base.values)
                    model.counts.copy_(base.counts)
                cpu_start, start = process_time_ns(), perf_counter_ns()
                new = torch.stack(
                    [
                        (
                            encoder.encode(episodes[i])
                            if method == "hdc"
                            else model_input_features(episodes[i])
                        )
                        for i in extra
                    ]
                )
                encoded = perf_counter_ns()
                if method == "hdc":
                    for vector, index in zip(new, extra):
                        model.update(vector, int(labels[index]))
                    diagnostics = None
                else:
                    training = torch.cat((features[selected], new), dim=0)
                    truth = torch.cat((labels[selected], labels[extra]))
                    model, diagnostics = fit_classifier(
                        method,
                        settings["parameters"][str(budget)][method],
                        seed,
                        training,
                        truth,
                        list(range(len(training))),
                    )
                learned = perf_counter_ns()
                predictions = (
                    [model.predict(vector) for vector in new]
                    if method == "hdc"
                    else model.predict(new.numpy()).tolist()
                )
                finished, cpu_finished = perf_counter_ns(), process_time_ns()
                # Correctness checks are outside measured stages.
                if len(predictions) != batch_size:
                    raise AssertionError("Update benchmark lost predictions")
                if method == "hdc" and (
                    not torch.equal(model.values, expected.values)
                    or not torch.equal(model.counts, expected.counts)
                ):
                    raise AssertionError(
                        "Timed additive updates disagree with the reviewed-memory rebuild"
                    )
                if repeat >= config.refit_warmup:
                    values = {
                        "encoding_ms": (encoded - start) / 1e6,
                        "learning_ms": (learned - encoded) / 1e6,
                        "prediction_ms": (finished - learned) / 1e6,
                        "complete_wall_ms": (finished - start) / 1e6,
                        "complete_cpu_ms": (cpu_finished - cpu_start) / 1e6,
                    }
                    for key, value in values.items():
                        samples[key].append(value)
                    if diagnostics is not None:
                        fit_diagnostics.append(diagnostics)
            rows.append(
                {
                    "method": method,
                    "batch_size": batch_size,
                    "initial_training_examples": len(selected),
                    "total_reviewed_examples": len(selected) + batch_size,
                    "initial_review_episode_ids": [episodes[i]["episode_id"] for i in selected],
                    "new_review_episode_ids": [episodes[i]["episode_id"] for i in extra],
                    "timings": {key: latency_summary(values) for key, values in samples.items()},
                    "per_new_sample": {
                        key: latency_summary([value / batch_size for value in values])
                        for key, values in samples.items()
                    },
                    "samples": samples,
                    "fit_diagnostics": fit_diagnostics,
                    "additive_memory_rebuild_exact": method == "hdc",
                }
            )
    return {
        "rows": rows,
        "skipped_batches": skipped,
        "warmup": config.refit_warmup,
        "repeats": config.refit_repeats,
        "scope": "Same initial reviews and chronological new memory reviews. Encode the arriving batch, incorporate its labels, then predict that batch. HDC adds each normalized hypervector; LR/MLP refit once on all retained reviewed features with frozen settings. Initial fitting and HDC state reset are excluded. Warm encoder cache, one CPU thread; wall and process CPU time recorded. Joins, eligibility, database work and persistence excluded.",
    }
