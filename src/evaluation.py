"""Comparable retrieval measurements and intervals over independent worlds."""

import math
import statistics
import time

import torch

from config import LABELS


def retrieval_metrics(rankings, query_indices, episodes, labels):
    rows = []
    for ranked, query in zip(rankings, query_indices):
        relevant = [int(labels[index]) == int(labels[query]) for index in ranked]
        rows.append(
            {
                "episode_id": episodes[query]["episode_id"],
                "world_id": episodes[query]["world_id"],
                "label": LABELS[int(labels[query])],
                "top1": float(relevant[0]) if relevant else 0.0,
                "precision_at_5": sum(relevant[:5]) / max(1, len(relevant[:5])),
                "reciprocal_rank_at_10": next(
                    (1 / (i + 1) for i, match in enumerate(relevant[:10]) if match), 0.0
                ),
                "ranked_episode_ids": [episodes[index]["episode_id"] for index in ranked],
            }
        )
    metrics = {
        key: statistics.mean(row[key] for row in rows)
        for key in ("top1", "precision_at_5", "reciprocal_rank_at_10")
    }
    per_world = {}
    for world in sorted({row["world_id"] for row in rows}):
        subset = [row for row in rows if row["world_id"] == world]
        per_world[world] = {key: statistics.mean(row[key] for row in subset) for key in metrics}
    return {"metrics": metrics, "per_world": per_world, "queries": rows}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def latency_summary(milliseconds):
    return {
        "median_ms": statistics.median(milliseconds),
        "p95_ms": percentile(milliseconds, 0.95),
        "n": len(milliseconds),
    }


def benchmark(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return latency_summary(values)


def grouped_interval(values_by_data_world, repeats=1000, seed=881):
    """Hierarchical resampling: data world sets, then complete worlds within a set.

    Encoder repetitions are averaged before entry. These intervals describe the
    controlled generator, not uncertainty across real telecom environments.
    """
    generator = torch.Generator().manual_seed(seed)
    groups = list(values_by_data_world.values())
    draws = []
    for _ in range(repeats):
        selected = torch.randint(len(groups), (len(groups),), generator=generator).tolist()
        values = []
        for index in selected:
            group = groups[index]
            worlds = torch.randint(len(group), (len(group),), generator=generator).tolist()
            values.extend(group[world] for world in worlds)
        draws.append(statistics.mean(values))
    return {
        "mean": statistics.mean(value for group in groups for value in group),
        "lower": percentile(draws, 0.025),
        "upper": percentile(draws, 0.975),
        "resampling": "Data seeds, then entire network/time blocks; encoder repeats averaged first",
    }
