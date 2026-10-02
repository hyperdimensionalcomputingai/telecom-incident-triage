"""Reproducible prepare / run / report stages for the telecom tutorial."""

import importlib.metadata
import json
import platform
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_info

from classifiers import METHODS, fit_classifier, select_settings
from config import BASE_TS, LABELS, ROOT, Config, digest, file_hash, save_json
from data import TABLES, build_episodes, generate, load_tables, source_lookup
from encoding import Encoder, Term, model_input_features, unit
from evaluation import benchmark, grouped_interval, latency_summary, retrieval_metrics
from fixtures import operator_diagnostics
from geography import load_geography, polygon_for
from learning import Prototype, budget_indices, delayed_feedback, learning_curves
from storage import SourceStore, VectorStore, directory_bytes
from update_cost import learning_update_cost

PACKAGES = (
    "numpy",
    "polars",
    "pyarrow",
    "geoarrow-pyarrow",
    "geodatafusion",
    "datafusion",
    "lancedb",
    "pylance",
    "torch",
    "torch-hd",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "joblib",
    "threadpoolctl",
)


def code_hash():
    # Editorial/figure revisions do not invalidate measured experiment results.
    paths = [
        *[
            path
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
            if path.name != "reporting.py"
        ],
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return digest({str(path.relative_to(ROOT)): file_hash(path) for path in paths})


def save_metrics(run_dir, results):
    """Retain measured evidence without bulky query traces or predictions."""
    pairs = []
    hashes = {}
    for result in results:
        pair = {
            key: value
            for key, value in result.items()
            if key not in ("retrieval", "learning", "operators")
        }
        pair["retrieval"] = {
            method: {key: value for key, value in row.items() if key != "queries"}
            for method, row in result["retrieval"].items()
        }
        pair["learning"] = [
            {key: value for key, value in row.items() if key != "predictions"}
            for row in result["learning"]
        ]
        pair["operators"] = {
            name: {
                key: value
                for key, value in row.items()
                if key not in ("before_episode", "after_episode")
            }
            if isinstance(row, dict)
            else row
            for name, row in result["operators"].items()
        }
        pairs.append(pair)
        relative = f"pairs/data-{result['data_seed']}_encoder-{result['encoder_seed']}/results.json"
        hashes[relative] = file_hash(Path(run_dir) / relative)
    save_json(
        Path(run_dir) / "metrics.json",
        {
            "scope": "Exact projection of result records; excludes retrieval query traces, individual predictions and full connectivity-fixture episodes. Resources include raw repeated timings.",
            "source_result_hashes": hashes,
            "pairs": pairs,
        },
    )


def prepare(run_dir, config):
    run_dir = Path(run_dir)
    config.validate()
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(
            "Run directory already exists; choose a new RUN_DIR in src/settings.py"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    load_geography()  # Verify the public input before copying its immutable run snapshot.
    geography_root = run_dir / "geography"
    geography_root.mkdir(exist_ok=True)
    for name in ("manifest.json", "bloor-corridor.geojson", "bloor-source-response.json"):
        shutil.copy2(ROOT / "data" / "geography" / name, geography_root / name)
    save_json(run_dir / "config.json", config.to_dict())
    manifest = {
        "stage": "preparing",
        "config_hash": digest(config.to_dict()),
        "prepare_code_hash": code_hash(),
        "datasets": {},
        "dataset_manifest_hashes": {},
        "prepare_timings_s": {},
        "geography_hashes": {
            path.name: file_hash(path) for path in geography_root.iterdir() if path.is_file()
        },
    }
    save_json(run_dir / "manifest.json", manifest)
    for seed in config.data_seeds:
        root = run_dir / "data" / str(seed)
        started = time.perf_counter()
        generate(root, config, seed, geography_root)
        manifest["prepare_timings_s"][str(seed)] = time.perf_counter() - started
        manifest["datasets"][str(seed)] = {
            name: file_hash(root / f"{name}.parquet") for name in TABLES
        }
        manifest["dataset_manifest_hashes"][str(seed)] = file_hash(root / "manifest.json")
        save_json(run_dir / "manifest.json", manifest)
        print(f"Prepared data seed {seed}", flush=True)
    manifest["stage"] = "prepared"
    save_json(run_dir / "manifest.json", manifest)
    return manifest


def verify_prepared(run_dir, config):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["config_hash"] != digest(config.to_dict()):
        raise ValueError("Prepared configuration hash mismatch")
    for name, checksum in manifest["geography_hashes"].items():
        if file_hash(run_dir / "geography" / name) != checksum:
            raise ValueError("Frozen public geography changed")
    for seed in config.data_seeds:
        if (
            file_hash(run_dir / "data" / str(seed) / "manifest.json")
            != manifest["dataset_manifest_hashes"][str(seed)]
        ):
            raise ValueError("Prepared dataset manifest changed")
        for name, checksum in manifest["datasets"][str(seed)].items():
            if file_hash(run_dir / "data" / str(seed) / f"{name}.parquet") != checksum:
                raise ValueError(f"Prepared source changed: {seed}/{name}")
    return manifest


def verify_evidence(query, candidate, encoder, lookup, raw_query, raw_candidate):
    """Reconstruct a candidate's raw bundle and attribute its cosine arithmetically.

    If candidate z = sum(t_j), cosine(q, z) = sum(unit(q) dot t_j / ||z||).
    Divide every term by the same complete candidate norm; normalizing terms
    separately would break the identity. Contributions can be negative and include
    interference: this decomposition accounts for a score, not a causal diagnosis.
    """
    manifest = encoder.manifest(candidate)
    terms = [Term(**record) for record in manifest["terms"]]
    reconstructed = encoder.encode_terms(terms)
    raw_error = float((reconstructed - raw_candidate).abs().max())
    query_unit = unit(raw_query)
    norm = torch.linalg.vector_norm(raw_candidate)
    contributions = []
    for term in terms:
        for identity in term.source_ids:
            if identity not in lookup:
                raise AssertionError(f"Unresolved evidence source: {identity}")
            source = lookup[identity]
            if source.get("available_s", 0) > candidate["decision_s"]:
                raise AssertionError("Future evidence source")
        contributions.append(
            {
                "component": term.component,
                "field": term.field,
                "position": term.position,
                "value": term.value,
                "source_ids": term.source_ids,
                "cosine_contribution": float(query_unit @ encoder.term_vector(term) / norm),
            }
        )
    score = float(query_unit @ unit(raw_candidate))
    contribution_error = abs(score - sum(row["cosine_contribution"] for row in contributions))
    if raw_error > 3e-6 or contribution_error > 2e-6:
        raise AssertionError("Evidence reconstruction mismatch")
    return {
        "query_episode_id": query["episode_id"],
        "candidate_episode_id": candidate["episode_id"],
        "raw_reconstruction_max_error": raw_error,
        "score_reconstruction_error": contribution_error,
        "float32_cosine": score,
        "contributions": contributions,
        "sources": [
            lookup[identity]
            for identity in sorted({identity for term in terms for identity in term.source_ids})
        ],
        "interpretation": "Exact arithmetic attribution including interference between terms; not causal attribution",
    }


def resource_measurements(
    encoder, episodes, raws, features, labels, memory_indices, config, store_path, settings, seed
):
    # Label prediction and review incorporation are separate operations for trained models.
    budget = max(config.budgets)
    selected = budget_indices(memory_indices, labels, budget, episodes)
    remaining = [
        i
        for i in memory_indices
        if i not in selected
        and episodes[i]["decision_s"] >= max(episodes[j]["decision_s"] for j in selected)
    ]
    if not remaining:
        raise ValueError(
            "Compute comparison needs one further memory review after its training budget"
        )
    extra = min(remaining, key=lambda i: (episodes[i]["decision_s"], episodes[i]["episode_id"]))
    episode, query, sample = episodes[extra], raws[extra], unit(raws[extra])
    accumulation = torch.zeros_like(sample)
    hdc = Prototype(encoder.dimension)
    for index in selected:
        hdc.update(raws[index], int(labels[index]))
    models, fit_diagnostics = {}, {}
    for method in METHODS:
        models[method], fit_diagnostics[method] = fit_classifier(
            method, settings["parameters"][str(budget)][method], seed, features, labels, selected
        )

    def complete_hdc():
        vector = encoder.encode(episode)
        hdc.predict(vector)
        hdc.update(vector, int(labels[extra]))

    def encode_predict(method):
        vector = model_input_features(episode)
        models[method].predict(vector.numpy().reshape(1, -1))

    def review_refit(method):
        # Retain earlier reviewed features; encode the new review, refit, then predict it.
        new_vector = model_input_features(episode)
        training = torch.cat((features[selected], new_vector[None, :]), dim=0)
        truth = torch.cat((labels[selected], labels[extra : extra + 1]))
        model, _ = fit_classifier(
            method,
            settings["parameters"][str(budget)][method],
            seed,
            training,
            truth,
            list(range(len(training))),
        )
        model.predict(new_vector.numpy().reshape(1, -1))

    measurements = {
        "hdc_encode": benchmark(
            lambda: encoder.encode(episode), config.warmup, config.benchmark_repeats
        ),
        "hdc_score": benchmark(lambda: hdc.predict(query), config.warmup, config.benchmark_repeats),
        "hdc_add_only": benchmark(
            lambda: accumulation.add_(sample), config.warmup, config.benchmark_repeats
        ),
        "hdc_encode_predict": benchmark(
            lambda: hdc.predict(encoder.encode(episode)), config.warmup, config.benchmark_repeats
        ),
        "hdc_encode_score_update": benchmark(complete_hdc, config.warmup, config.benchmark_repeats),
    }
    for method in METHODS:
        measurements[method + "_encode_predict"] = benchmark(
            lambda method=method: encode_predict(method), config.warmup, config.benchmark_repeats
        )
        measurements[method + "_review_refit"] = benchmark(
            lambda method=method: review_refit(method), config.refit_warmup, config.refit_repeats
        )
    return {
        "timings": measurements,
        "scope": "Warm CPU batch-one prediction; HDC additive update versus batch LR/MLP refit from the same retained reviews. Joins, geography, audit persistence and database work excluded.",
        "review_refit_training_examples": len(selected) + 1,
        "review_buffer_feature_and_label_bytes": (len(selected) + 1)
        * (features.shape[1] * 4 + labels.element_size()),
        "fit_diagnostics": fit_diagnostics,
        "hdc_raw_vector_bytes": encoder.dimension * 4,
        "hdc_search_vector_bytes": encoder.dimension * 2,
        "model_input_bytes": features.shape[1] * 4,
        "class_memory_bytes": {"hdc": len(LABELS) * encoder.dimension * 4},
        "vector_store_bytes": directory_bytes(store_path),
        "hdc_representation_tensor_bytes": raws.numel() * raws.element_size(),
        "model_input_tensor_bytes": features.numel() * 4,
        "encoder_cached_basis_bytes": encoder.basis.cache_info().currsize * encoder.dimension * 4,
        "not_measured": [
            "Energy",
            "GPU",
            "Distributed serving",
            "Production throughput",
            "Peak process memory",
            "Incremental LR or MLP optimizers",
        ],
    }


def evaluate_pair(
    run_dir,
    dataset_seed,
    encoder_seed,
    config,
    tables,
    episodes,
    labels,
    reviews,
    lookup,
    candidates,
    setting,
    gate_timing,
    source_bytes,
    dataset_timings,
):
    root = run_dir / "pairs" / f"data-{dataset_seed}_encoder-{encoder_seed}"
    root.mkdir(parents=True, exist_ok=True)
    encoder = Encoder(config, encoder_seed)
    signature = encoder.signature()
    start = time.perf_counter()
    raws = torch.stack([encoder.encode(episode) for episode in episodes])
    encode_s = time.perf_counter() - start
    # Each row is one complete episode. Normalize across its D coordinates (dim=1),
    # preserving raw sums for reconstruction and component edits.
    vectors = F.normalize(raws, dim=1)
    features = torch.stack([model_input_features(episode) for episode in episodes])
    memory = [i for i, episode in enumerate(episodes) if episode["split"] == "memory"]
    testing = [i for i, episode in enumerate(episodes) if episode["split"] == "test"]
    by_id = {episode["episode_id"]: i for i, episode in enumerate(episodes)}
    start = time.perf_counter()
    store = VectorStore(root / "vectors", episodes, raws, encoder)
    persistence_s = time.perf_counter() - start
    rank_sets, times = {}, {}
    for method in (
        "hdc",
        "hdc_without_paths",
        "hdc_without_order",
        "hdc_lance_float16",
    ):
        if method.startswith("hdc_without"):
            kwargs = {"path": False} if method.endswith("paths") else {"sequence": False}
            x = F.normalize(
                torch.stack([encoder.encode(episode, **kwargs) for episode in episodes]), dim=1
            )
        else:
            x = vectors
        rankings, durations = [], []
        for query in testing:
            ids = candidates[episodes[query]["episode_id"]]
            indices = torch.tensor([by_id[identity] for identity in ids], dtype=torch.int64)
            if not len(indices):
                raise ValueError("Final query has no eligible earlier memory episodes")
            start = time.perf_counter_ns()
            if method == "hdc_lance_float16":
                hits = store.search(raws[query], ids, episodes[query]["decision_s"])
                order = [by_id[hit["episode_id"]] for hit in hits]
                duration = (time.perf_counter_ns() - start) / 1e6
                expected = (
                    (store.quantized[indices] @ vectors[query]).sort(descending=True).values[:10]
                )
                actual = torch.tensor([1 - hit["_distance"] for hit in hits])
                if not torch.allclose(actual, expected, atol=2e-5, rtol=1e-5):
                    raise AssertionError(
                        "Stored-vector retrieval disagrees with exhaustive float32 verification"
                    )
            else:
                scores = x[indices] @ x[query]
                order = indices[torch.argsort(scores, descending=True, stable=True)[:10]].tolist()
                duration = (time.perf_counter_ns() - start) / 1e6
            rankings.append(order)
            durations.append(duration)
        rank_sets[method] = retrieval_metrics(rankings, testing, episodes, labels)
        times[method] = latency_summary(durations)
    random_precision = statistics.mean(
        sum(
            int(labels[by_id[identity]]) == int(labels[query])
            for identity in candidates[episodes[query]["episode_id"]]
        )
        / len(candidates[episodes[query]["episode_id"]])
        for query in testing
    )
    curves = learning_curves(
        raws,
        features,
        labels,
        memory,
        testing,
        episodes,
        config,
        setting,
        encoder_seed,
        root / "models",
    )
    online = delayed_feedback(raws, labels, memory, episodes, reviews, config, signature)
    if encoder.signature() != signature:
        raise AssertionError("Encoder changed during online learning")
    operators = operator_diagnostics(encoder, tables, episodes, reviews)
    candidate_counts = [len(candidates[episodes[query]["episode_id"]]) for query in testing]
    # Every retained top hit is checked, rather than auditing only the displayed success case.
    max_raw_error, max_score_error = 0.0, 0.0
    evidence_by_query = {}
    for row, query in zip(rank_sets["hdc_lance_float16"]["queries"], testing):
        candidate = by_id[row["ranked_episode_ids"][0]]
        evidence = verify_evidence(
            episodes[query], episodes[candidate], encoder, lookup, raws[query], raws[candidate]
        )
        stored_score = float(vectors[query] @ store.quantized[candidate])
        evidence["stored_float16_cosine"] = stored_score
        evidence["quantization_residual"] = stored_score - evidence["float32_cosine"]
        evidence["score_scope"] = (
            "Term contributions reconstruct float32 cosine; add the recorded quantization residual for the stored-vector cosine."
        )
        max_raw_error = max(max_raw_error, evidence["raw_reconstruction_max_error"])
        max_score_error = max(max_score_error, evidence["score_reconstruction_error"])
        evidence_by_query[episodes[query]["episode_id"]] = evidence
    # The observable disruption motivates the analyst query; no correctness score selects it.
    query = next(
        index
        for index in testing
        if any(row["service_disruption"] for row in episodes[index]["observations"])
    )
    query_position = testing.index(query)
    candidate = by_id[
        rank_sets["hdc_lance_float16"]["queries"][query_position]["ranked_episode_ids"][0]
    ]
    stripped = raws[query] - encoder.components(episodes[query])["handset"]
    edited_hits = store.search(
        stripped, candidates[episodes[query]["episode_id"]], episodes[query]["decision_s"], 5
    )
    case = {
        "selection": "First final-test episode with an observed service disruption; selected without inspecting retrieval correctness",
        "query": episodes[query],
        "query_label": LABELS[int(labels[query])],
        "candidate": episodes[candidate],
        "candidate_label": LABELS[int(labels[candidate])],
        "evidence": evidence_by_query[episodes[query]["episode_id"]],
        "original_top5": rank_sets["hdc_lance_float16"]["queries"][query_position][
            "ranked_episode_ids"
        ][:5],
        "without_handset_top5": [hit["episode_id"] for hit in edited_hits],
        "query_edit_max_error": float(
            (stripped - encoder.encode(episodes[query], handset=False)).abs().max()
        ),
    }
    errors = [
        dict(row, retrieved_label=reviews[row["ranked_episode_ids"][0]]["label"])
        for row in rank_sets["hdc_lance_float16"]["queries"]
        if not row["top1"]
    ]
    resources = resource_measurements(
        encoder,
        episodes,
        raws,
        features,
        labels,
        memory,
        config,
        root / "vectors",
        setting,
        encoder_seed,
    )
    resources.update(
        encode_all_s=encode_s,
        persistence_s=persistence_s,
        exact_gate_s=gate_timing,
        source_store_bytes=source_bytes,
        retrieval_latency=times,
    )
    resources["learning_updates"] = learning_update_cost(
        encoder, episodes, raws, features, labels, memory, config, setting, encoder_seed
    )
    resources["dataset_timings"] = dataset_timings
    result = {
        "data_seed": dataset_seed,
        "encoder_seed": encoder_seed,
        "encoder_hash": signature,
        "code_hash": code_hash(),
        "config_hash": digest(config.to_dict()),
        "n_memory": len(memory),
        "n_queries": len(testing),
        "candidate_counts": {
            "min": min(candidate_counts),
            "mean": statistics.mean(candidate_counts),
            "max": max(candidate_counts),
        },
        "random_precision_at_5": random_precision,
        "retrieval": rank_sets,
        "learning": curves,
        "online_metrics": online["metrics"],
        "operators": operators,
        "evidence_audit": {
            "top_hits_checked": len(testing),
            "source_resolution": 1.0,
            "max_raw_reconstruction_error": max_raw_error,
            "max_score_reconstruction_error": max_score_error,
        },
        "float16_quantization": {
            "max_coordinate_error": float((store.quantized - vectors).abs().max()),
            "top1_rank_changes": sum(
                a["ranked_episode_ids"][0] != b["ranked_episode_ids"][0]
                for a, b in zip(
                    rank_sets["hdc"]["queries"], rank_sets["hdc_lance_float16"]["queries"]
                )
            ),
            "precision_at_5_change": rank_sets["hdc_lance_float16"]["metrics"]["precision_at_5"]
            - rank_sets["hdc"]["metrics"]["precision_at_5"],
        },
        "resources": resources,
    }
    save_json(root / "results.json", result)
    save_json(root / "case.json", case)
    save_json(root / "retrieval_errors.json", errors)
    save_json(root / "online.json", online)
    return result


def summarize(results, config):
    output = {"retrieval": {}, "learning": {}, "paired_differences": {}, "learning_differences": {}}
    for method in results[0]["retrieval"]:
        output["retrieval"][method] = {}
        for metric in ("top1", "precision_at_5", "reciprocal_rank_at_10"):
            worlds = defaultdict(lambda: defaultdict(list))
            for result in results:
                for world, values in result["retrieval"][method]["per_world"].items():
                    worlds[result["data_seed"]][world].append(values[metric])
            groups = {
                seed: [statistics.mean(values) for values in by_world.values()]
                for seed, by_world in worlds.items()
            }
            interval = grouped_interval(groups, config.bootstrap_repeats)
            means = [result["retrieval"][method]["metrics"][metric] for result in results]
            output["retrieval"][method][metric] = dict(
                interval, minimum_run=min(means), maximum_run=max(means)
            )
    for budget in config.budgets:
        output["learning"][str(budget)] = {}
        for method in ("hdc", *METHODS):
            worlds = defaultdict(lambda: defaultdict(list))
            for result in results:
                row = next(
                    row
                    for row in result["learning"]
                    if row["method"] == method and row["budget_per_class"] == budget
                )
                for world, value in row["per_world"].items():
                    worlds[result["data_seed"]][world].append(value["macro_f1"])
            output["learning"][str(budget)][method] = grouped_interval(
                {
                    seed: [statistics.mean(v) for v in group.values()]
                    for seed, group in worlds.items()
                },
                config.bootstrap_repeats,
            )
        output["learning_differences"][str(budget)] = {}
        for method in METHODS:
            worlds = defaultdict(lambda: defaultdict(list))
            for result in results:
                hdc = next(
                    row
                    for row in result["learning"]
                    if row["method"] == "hdc" and row["budget_per_class"] == budget
                )
                baseline = next(
                    row
                    for row in result["learning"]
                    if row["method"] == method and row["budget_per_class"] == budget
                )
                for world, value in hdc["per_world"].items():
                    worlds[result["data_seed"]][world].append(
                        value["macro_f1"] - baseline["per_world"][world]["macro_f1"]
                    )
            output["learning_differences"][str(budget)]["hdc_minus_" + method] = grouped_interval(
                {
                    seed: [statistics.mean(v) for v in group.values()]
                    for seed, group in worlds.items()
                },
                config.bootstrap_repeats,
            )
    for other in ("hdc_without_paths", "hdc_without_order"):
        worlds = defaultdict(lambda: defaultdict(list))
        for result in results:
            for world, values in result["retrieval"]["hdc"]["per_world"].items():
                worlds[result["data_seed"]][world].append(
                    values["precision_at_5"]
                    - result["retrieval"][other]["per_world"][world]["precision_at_5"]
                )
        output["paired_differences"]["hdc_minus_" + other] = grouped_interval(
            {seed: [statistics.mean(v) for v in group.values()] for seed, group in worlds.items()},
            config.bootstrap_repeats,
        )
    output["random_precision_at_5"] = statistics.mean(
        result["random_precision_at_5"] for result in results
    )
    output["n_seed_pairs"] = len(results)
    output["final_queries_scored"] = sum(result["n_queries"] for result in results)
    output["independent_final_worlds"] = (
        config.blocks - config.memory_blocks - config.validation_blocks
    ) * len(config.data_seeds)
    return output


def run(run_dir):
    run_dir = Path(run_dir)
    config = Config.read(run_dir / "config.json")
    torch.set_num_threads(1)
    manifest = verify_prepared(run_dir, config)
    manifest.update(
        stage="running",
        run_code_hash=code_hash(),
        completed_pairs=[],
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": 1,
            "threadpools": threadpool_info(),
            "versions": {package: importlib.metadata.version(package) for package in PACKAGES},
        },
    )
    save_json(run_dir / "manifest.json", manifest)
    results, setting = [], None
    for seed in config.data_seeds:
        tables = load_tables(run_dir / "data" / str(seed))
        started = time.perf_counter()
        episodes = build_episodes(tables)
        join_s = time.perf_counter() - started
        reviews = {row["episode_id"]: row for row in tables["reviews"].to_pylist()}
        labels = torch.tensor([LABELS.index(reviews[e["episode_id"]]["label"]) for e in episodes])
        lookup = source_lookup(tables)
        features = torch.stack([model_input_features(episode) for episode in episodes])
        memory = [i for i, episode in enumerate(episodes) if episode["split"] == "memory"]
        validation = [i for i, episode in enumerate(episodes) if episode["split"] == "validation"]
        if setting is None:
            setting = select_settings(features, labels, memory, validation, episodes, config)
            save_json(run_dir / "frozen_settings.json", setting)
        source_root = run_dir / "data" / str(seed) / "lance"
        started = time.perf_counter()
        sources = SourceStore(source_root, tables)
        source_write_s = time.perf_counter() - started
        candidates = {}
        start = time.perf_counter()
        for episode in episodes:
            if episode["split"] in ("validation", "test"):
                candidates[episode["episode_id"]] = sources.eligible(
                    polygon_for(episode), BASE_TS, episode["decision_s"], crosscheck=True
                )
        gate_timing = time.perf_counter() - start
        save_json(run_dir / "data" / str(seed) / "eligibility.json", candidates)
        print(f"Eligibility verified for data seed {seed}", flush=True)
        for encoder_seed in config.encoder_seeds:
            result = evaluate_pair(
                run_dir,
                seed,
                encoder_seed,
                config,
                tables,
                episodes,
                labels,
                reviews,
                lookup,
                candidates,
                setting,
                gate_timing,
                directory_bytes(source_root),
                {
                    "generation_and_parquet_s": manifest["prepare_timings_s"][str(seed)],
                    "factual_joins_and_source_validation_s": join_s,
                    "source_lance_persistence_s": source_write_s,
                },
            )
            results.append(result)
            manifest["completed_pairs"].append([seed, encoder_seed])
            save_json(run_dir / "manifest.json", manifest)
            print(
                f"Completed data {seed}, encoder {encoder_seed}: P@5={result['retrieval']['hdc']['metrics']['precision_at_5']:.3f}",
                flush=True,
            )
    summary = summarize(results, config)
    save_json(run_dir / "summary.json", summary)
    save_metrics(run_dir, results)
    manifest.update(
        stage="complete",
        summary_hash=file_hash(run_dir / "summary.json"),
        result_hashes={
            f"{result['data_seed']}/{result['encoder_seed']}": file_hash(
                run_dir
                / "pairs"
                / f"data-{result['data_seed']}_encoder-{result['encoder_seed']}"
                / "results.json"
            )
            for result in results
        },
    )
    artifacts = [run_dir / "frozen_settings.json", run_dir / "metrics.json"]
    for result in results:
        pair = run_dir / "pairs" / f"data-{result['data_seed']}_encoder-{result['encoder_seed']}"
        artifacts.extend(
            pair / name
            for name in ("results.json", "case.json", "online.json", "retrieval_errors.json")
        )
        artifacts.extend(sorted((pair / "models").glob("*.joblib")))
    manifest["artifact_hashes"] = {
        str(path.relative_to(run_dir)): file_hash(path) for path in artifacts
    }
    save_json(run_dir / "manifest.json", manifest)
    return summary


def report(run_dir):
    from reporting import create_report

    return create_report(Path(run_dir))
