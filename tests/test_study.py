"""Acceptance tests exercise the scientific and storage boundaries, not frozen scores."""

import json
from copy import deepcopy
from dataclasses import replace

import pyarrow as pa
import pytest
import torch

from config import BASE_TS, LABELS, Config, digest
from data import build_episodes, generate, source_lookup
from encoding import Encoder, Term, explicit_features, unit
from fixtures import operator_diagnostics
from geography import load_geography, polygon_for
from learning import Prototype, delayed_feedback, learning_curves
from oracle import derive_reviews
from storage import SourceStore, VectorStore, predicate


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("study")
    config = replace(
        Config(),
        data_seeds=(1001,),
        encoder_seeds=(2001,),
        blocks=3,
        memory_blocks=1,
        validation_blocks=1,
        episodes_per_block=24,
        budgets=(1, 2, 5),
        bootstrap_repeats=50,
        benchmark_repeats=10,
        warmup=2,
    )
    tables = generate(root / "data", config, 1001)
    episodes = build_episodes(tables)
    encoder = Encoder(config, 2001)
    raws = torch.stack([encoder.encode(episode) for episode in episodes])
    reviews = {row["episode_id"]: row for row in tables["reviews"].to_pylist()}
    labels = torch.tensor(
        [LABELS.index(reviews[episode["episode_id"]]["label"]) for episode in episodes]
    )
    sources = SourceStore(root / "sources", tables)
    vectors = VectorStore(root / "vectors", episodes, raws, encoder)
    return root, config, tables, episodes, encoder, raws, reviews, labels, sources, vectors


def amend(tables, name, change):
    rows = deepcopy(tables[name].to_pylist())
    change(rows)
    geometry = tables[name]["geometry"] if "geometry" in tables[name].column_names else None
    for row in rows:
        row.pop("geometry", None)
        row.pop("source_sha256")
        row["source_sha256"] = digest(row)
    schema = (
        tables[name].schema.remove(tables[name].schema.get_field_index("geometry"))
        if geometry is not None
        else tables[name].schema
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    if geometry is not None:
        table = table.append_column("geometry", geometry)
    return dict(tables, **{name: table})


def test_counts_and_all_classes_in_each_partition(sample):
    _, config, tables, episodes, _, _, reviews, *_ = sample
    assert len(episodes) == config.blocks * config.episodes_per_block
    assert tables["observations"].num_rows == len(episodes) * 3
    for partition in ("memory", "validation", "test"):
        assert {
            reviews[e["episode_id"]]["label"] for e in episodes if e["split"] == partition
        } == set(LABELS)


def test_disjoint_worlds_subscribers_and_incidents(sample):
    episodes = sample[3]
    for field in ("world_id", "subscriber_id", "phone_id", "episode_id"):
        groups = {
            partition: {e[field] for e in episodes if e["split"] == partition}
            for partition in ("memory", "validation", "test")
        }
        assert not groups["memory"] & groups["validation"]
        assert not groups["memory"] & groups["test"]
        assert not groups["validation"] & groups["test"]
    assert max(e["decision_s"] for e in episodes if e["split"] == "memory") < min(
        e["decision_s"] for e in episodes if e["split"] == "test"
    )


def test_shared_global_telemetry_and_time_valid_edges(sample):
    _, _, tables, episodes, *_ = sample
    for table, entity in (("cell_status", "cell_id"), ("link_status", "link_id")):
        records = tables[table].to_pylist()
        assert len({(r[entity], r["timestamp_s"]) for r in records}) == len(records)
    for episode in episodes:
        for row in episode["observations"]:
            assert row["valid_from_s"] <= row["timestamp_s"] < row["valid_to_s"]
            peers = [
                p
                for p in tables["peer_observations"].to_pylist()
                if p["source_id"] in row["peer_source_ids"]
            ]
            assert len(peers) == 2
            assert abs(sum(p["peer_loss_pct"] for p in peers) / 2 - row["link_loss_pct"]) < 0.021


def test_independent_oracle_reconstructs_labels(sample):
    _, config, tables, _, _, _, reviews, *_ = sample
    rows = {
        name: table.drop(["geometry"]).to_pylist()
        if "geometry" in table.column_names
        else table.to_pylist()
        for name, table in tables.items()
    }
    rows["episodes"] = list(reversed(rows["episodes"]))
    reconstructed = derive_reviews(rows, config.review_delay_s)
    assert {r["episode_id"]: r["label"] for r in reconstructed} == {
        key: r["label"] for key, r in reviews.items()
    }


def test_missing_and_duplicate_factual_records_fail(sample):
    tables = sample[2]
    missing = amend(tables, "link_status", lambda rows: rows.pop(0))
    with pytest.raises(ValueError, match="Missing"):
        build_episodes(missing)
    duplicate = amend(tables, "cell_status", lambda rows: rows.append(deepcopy(rows[0])))
    with pytest.raises(ValueError, match="Duplicate"):
        build_episodes(duplicate)


def test_ambiguous_telemetry_join_fails(sample):
    def insert(rows):
        row = deepcopy(rows[0])
        row["source_id"] += "-duplicate"
        rows.append(row)

    bad = amend(sample[2], "cell_status", insert)
    with pytest.raises(Exception, match="validation|unique"):
        build_episodes(bad)


def test_stale_topology_fails(sample):
    bad = amend(sample[2], "edges", lambda rows: rows[0].update(valid_to_s=rows[0]["valid_from_s"]))
    with pytest.raises(ValueError, match="stale"):
        build_episodes(bad)


def test_future_telemetry_and_peer_information_fail(sample):
    for name in ("link_status", "peer_observations"):
        bad = amend(
            sample[2], name, lambda rows: rows[0].update(available_s=rows[0]["timestamp_s"] + 99999)
        )
        with pytest.raises(ValueError, match="future"):
            build_episodes(bad)


def test_partial_or_duplicate_episode_observations_fail(sample):
    bad = amend(
        sample[2],
        "episodes",
        lambda rows: rows[0].update(observation_ids=rows[0]["observation_ids"][:2]),
    )
    # Metadata must match actual observation identities, not merely an inferred count.
    with pytest.raises(ValueError, match="identit|observation"):
        build_episodes(bad)


def test_hash_mismatch_fails(sample):
    tables = dict(sample[2])
    rows = tables["phones"].to_pylist()
    rows[0]["handset"] = "tampered"
    tables["phones"] = pa.Table.from_pylist(rows, schema=tables["phones"].schema)
    with pytest.raises(ValueError, match="checksum"):
        source_lookup(tables)


def test_correlated_numeric_levels(sample):
    encoder = sample[4]
    adjacent = float(unit(encoder.levels[0]) @ unit(encoder.levels[1]))
    far = float(unit(encoder.levels[0]) @ unit(encoder.levels[-1]))
    assert adjacent > 0.8 and adjacent > far


def test_operator_invariances_and_counterfactual_edges(sample):
    _, _, tables, episodes, encoder, _, reviews, *_ = sample
    operators = operator_diagnostics(encoder, tables, episodes, reviews)
    assert operators["role_binding"]["bound_cosine"] < 0.2
    assert operators["role_binding"]["without_binding_max_error"] == 0
    assert operators["role_binding"]["unbinding_exact"]
    assert operators["event_order"]["with_order_cosine"] < 0.999
    assert operators["event_order"]["without_order_max_error"] < 3e-6
    assert operators["connectivity"]["with_path_cosine"] < 0.999
    assert operators["connectivity"]["without_path_max_error"] < 3e-6
    assert operators["connectivity"]["correct_pair_rank"]


def test_identifiers_labels_and_outcomes_are_excluded(sample):
    query = deepcopy(sample[3][0])
    encoder = sample[4]
    before = encoder.encode(query)
    query.update(
        episode_id="renamed",
        subscriber_id="renamed",
        phone_id="renamed",
        world_id="renamed",
        label="invented",
        split="test",
        decision_s=-999,
    )
    for row in query["observations"]:
        for key in ("cell_id", "link_id", "phone_id", "source_id", "edge_id"):
            row[key] = "renamed"
        row.update(
            service_disruption=not row["service_disruption"],
            longitude=0.0,
            latitude=0.0,
            timestamp_s=-999,
        )
    torch.testing.assert_close(before, encoder.encode(query), rtol=0, atol=0)


def test_raw_query_edit_and_manifest_reconstruction(sample):
    _, _, _, episodes, encoder, raws, *_ = sample
    query = episodes[0]
    handset = encoder.components(query)["handset"]
    torch.testing.assert_close(
        raws[0] - handset, encoder.encode(query, handset=False), atol=3e-6, rtol=0
    )
    manifest = encoder.manifest(query)
    reconstructed = encoder.encode_terms([Term(**term) for term in manifest["terms"]])
    torch.testing.assert_close(raws[0], reconstructed, atol=0, rtol=0)


def test_stored_manifests_and_edge_sources_resolve(sample):
    _, _, tables, _, encoder, _, _, _, _, vectors = sample
    lookup = source_lookup(tables)
    row = vectors.table.to_arrow().slice(0, 1).to_pylist()[0]
    manifest = json.loads(row["manifest_json"])
    rebuilt = encoder.encode_terms([Term(**term) for term in manifest["terms"]])
    torch.testing.assert_close(rebuilt, torch.tensor(row["raw_vector"]), atol=0, rtol=0)
    assert all(identity in lookup for identity in row["source_ids"])
    assert any(
        identity in {e["source_id"] for e in tables["edges"].to_pylist()}
        for identity in row["source_ids"]
    )


def test_geoarrow_metadata_survives_roundtrip(sample):
    field = sample[8].observations.schema.field("geometry")
    assert field.type.extension_name == "geoarrow.wkb"
    assert any(name in str(field.type.crs) for name in ("EPSG:4326", "OGC:CRS84"))
    assert sample[9].table.schema.field("vector").type.value_type == pa.float16()
    assert sample[9].table.schema.field("raw_vector").type.value_type == pa.float32()


def test_exact_geo_time_and_earlier_memory_gate(sample):
    _, _, _, episodes, _, raws, _, _, sources, vectors = sample
    query = episodes[-1]
    ids = sources.eligible(polygon_for(query), BASE_TS, query["decision_s"])
    hits = vectors.search(raws[-1], ids, query["decision_s"])
    by_id = {e["episode_id"]: e for e in episodes}
    assert hits
    assert all(
        by_id[hit["episode_id"]]["split"] == "memory"
        and by_id[hit["episode_id"]]["decision_s"] < query["decision_s"]
        for hit in hits
    )


def test_geo_boundaries_partial_episode_and_empty_candidates(sample):
    episodes, sources, vectors = sample[3], sample[8], sample[9]
    query = episodes[0]
    start = query["observations"][0]["timestamp_s"]
    assert query["episode_id"] not in sources.eligible(polygon_for(query), start, start + 20)
    assert query["episode_id"] in sources.eligible(polygon_for(query), start, start + 40)
    sql = "SELECT ST_Intersects(ST_GeomFromText('POINT (0 0)'), ST_GeomFromText('POLYGON ((0 0,1 0,1 1,0 1,0 0))')) AS v"
    assert sources.ctx.sql(sql).to_arrow_table()["v"][0].as_py() is True
    assert sources.eligible("POLYGON ((0 0,1 0,1 1,0 1,0 0))", BASE_TS, query["decision_s"]) == []
    assert vectors.search(sample[5][0], [], query["decision_s"]) == []


def test_exact_search_verifies_quantized_vectors(sample):
    _, _, _, episodes, _, raws, _, _, sources, vectors = sample
    query = episodes[-1]
    ids = sources.eligible(polygon_for(query), BASE_TS, query["decision_s"])
    by_id = {episode["episode_id"]: i for i, episode in enumerate(episodes)}
    expected = (
        (vectors.quantized[[by_id[identity] for identity in ids]] @ unit(raws[-1]))
        .sort(descending=True)
        .values[:10]
    )
    hits = vectors.search(raws[-1], ids, query["decision_s"])
    actual = torch.tensor([1 - hit["_distance"] for hit in hits])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def test_empty_memory_and_exact_reversal():
    model = Prototype(256)
    vector = torch.randn(256, generator=torch.Generator().manual_seed(4))
    assert model.predict(vector) is None
    before = model.values.clone()
    undo = model.update(vector, 2, {"episode_id": "reviewed"})
    assert model.predict(vector) == 2
    model.undo(undo)
    assert torch.equal(before, model.values) and not model.history and int(model.counts.sum()) == 0


def test_delayed_reviews_do_not_leak_test_labels(sample):
    _, config, _, episodes, encoder, raws, reviews, labels, *_ = sample
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    signature = encoder.signature()
    online = delayed_feedback(raws, labels, memory, episodes, reviews, config, signature)
    assert online["predictions"]["hdc"][0]["prediction"] is None
    assert online["predictions"]["hdc"][0]["reviews_available"] == 0
    assert len(online["updates"]) == len(memory)
    assert {row["episode_id"] for row in online["updates"]} == {
        episodes[i]["episode_id"] for i in memory
    }
    assert all(row["review_ready_s"] <= row["observed_at_s"] for row in online["updates"])
    assert encoder.signature() == signature and online["reversal_exact"]
    for prediction in online["predictions"]["hdc"]:
        earlier = sum(row["observed_at_s"] <= prediction["decision_s"] for row in online["updates"])
        assert earlier == prediction["reviews_available"]


def test_matched_label_budgets_and_held_out_curves(sample):
    _, config, _, episodes, _, raws, _, labels, *_ = sample
    features = torch.stack([explicit_features(e) for e in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    testing = [i for i, e in enumerate(episodes) if e["split"] == "test"]
    settings = {
        "parameters": {
            str(b): {"logistic_regression": {"C": 1.0}, "mlp": {"width": 16, "alpha": 0.1}}
            for b in config.budgets
        }
    }
    rows = learning_curves(raws, features, labels, memory, testing, episodes, config, settings)
    for budget in config.budgets:
        subset = [row for row in rows if row["budget_per_class"] == budget]
        assert len(subset) == 3
        assert all(len(row["review_episode_ids"]) == 4 * budget for row in subset)
        assert len({tuple(row["review_episode_ids"]) for row in subset}) == 1
        assert all(
            set(row["review_episode_ids"]).isdisjoint({episodes[i]["episode_id"] for i in testing})
            for row in subset
        )


def test_classifier_scaler_and_training_use_reviewed_examples_only(sample, tmp_path):
    from classifiers import fit_classifier, persist_classifier
    from learning import budget_indices

    _, _, _, episodes, _, _, _, labels, *_ = sample
    features = torch.stack([explicit_features(e) for e in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    selected = budget_indices(memory, labels, 2, episodes)
    for method, parameters in [
        ("logistic_regression", {"C": 1.0}),
        ("mlp", {"width": 16, "alpha": 0.1}),
    ]:
        model, info = fit_classifier(method, parameters, 2001, features, labels, selected)
        assert torch.allclose(torch.as_tensor(model[0].mean_).float(), features[selected].mean(0))
        assert int(model[0].n_samples_seen_) == 8
        assert model[-1].max_iter == 3000
        assert not info["convergence_warnings"]
        persisted = persist_classifier(model, features, tmp_path / f"{method}.joblib")
        assert persisted["prediction_roundtrip_exact"] and persisted["bytes"] > 0


def test_model_selection_rejects_test_partition(sample):
    from classifiers import select_settings

    _, config, _, episodes, _, _, _, labels, *_ = sample
    features = torch.stack([explicit_features(e) for e in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    testing = [i for i, e in enumerate(episodes) if e["split"] == "test"]
    with pytest.raises(ValueError, match="validation"):
        select_settings(features, labels, memory, testing, episodes, config)


def test_classifiers_ignore_labels_outside_training(sample):
    from classifiers import fit_classifier
    from learning import budget_indices

    _, _, _, episodes, _, _, _, labels, *_ = sample
    features = torch.stack([explicit_features(e) for e in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    selected = budget_indices(memory, labels, 2, episodes)
    changed = labels.clone()
    for i in range(len(episodes)):
        if i not in selected:
            changed[i] = (labels[i] + 1) % 4
    for method, parameters in [
        ("logistic_regression", {"C": 1.0}),
        ("mlp", {"width": 16, "alpha": 0.1}),
    ]:
        first, _ = fit_classifier(method, parameters, 2001, features, labels, selected)
        second, _ = fit_classifier(method, parameters, 2001, features, changed, selected)
        assert first.predict(features.numpy()).tolist() == second.predict(features.numpy()).tolist()


def test_public_snapshot_checksums_and_safe_predicate():
    lines, manifest = load_geography()
    assert len(lines) > 10 and manifest["corridor_segments"] > 10
    assert manifest["source_sha256"] and manifest["corridor_sha256"]
    with pytest.raises(ValueError, match="Unsafe"):
        predicate("POLYGON ('; DROP TABLE x)", 0, 1)


def test_config_rejects_invalid_partitions():
    with pytest.raises(ValueError, match="Final"):
        replace(Config(), blocks=18).validate()
    with pytest.raises(ValueError, match="coverage"):
        replace(Config(), episodes_per_block=99).validate()


def test_coordinate_system_mismatch_is_rejected(sample, tmp_path):
    import geoarrow.pyarrow as ga

    tables = dict(sample[2])
    original = tables["observations"]
    geometry = (
        ga.wkb().with_crs("EPSG:3857").wrap_array(original["geometry"].combine_chunks().storage)
    )
    tables["observations"] = original.drop(["geometry"]).append_column("geometry", geometry)
    with pytest.raises(ValueError, match="CRS"):
        SourceStore(tmp_path / "bad-crs", tables)


def test_persisted_audit_log_can_replay_and_restore_a_prefix(sample):
    from learning import replay_updates, tensor_hash

    _, config, _, episodes, encoder, raws, reviews, labels, *_ = sample
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    online = delayed_feedback(raws, labels, memory, episodes, reviews, config, encoder.signature())
    raw_by_id = {episode["episode_id"]: raws[i] for i, episode in enumerate(episodes)}
    final = replay_updates(raw_by_id, online["updates"], config.dimension)
    previous = replay_updates(raw_by_id, online["updates"][:-1], config.dimension)
    last = online["updates"][-1]
    label = LABELS.index(last["label"])
    assert tensor_hash(final.values[label]) == last["updated_hash"]
    assert tensor_hash(previous.values[label]) == last["previous_hash"]
    assert int(final.counts.sum()) == len(memory)


def test_score_contributions_reconstruct_similarity(sample):
    from study import verify_evidence

    _, _, tables, episodes, encoder, raws, *_ = sample
    result = verify_evidence(
        episodes[-1], episodes[0], encoder, source_lookup(tables), raws[-1], raws[0]
    )
    assert result["raw_reconstruction_max_error"] < 3e-6
    assert result["score_reconstruction_error"] < 2e-6
    assert result["sources"] and result["contributions"]


def test_torchhd_bundling_preserves_raw_accumulator(sample):
    import torchhd

    from encoding import terms_for

    encoder, episode = sample[4], sample[3][0]
    terms = terms_for(episode, sample[1])
    values = torch.stack([encoder.term_vector(term) for term in terms])
    torch.testing.assert_close(encoder.encode(episode), values.sum(0), atol=0, rtol=0)
    torch.testing.assert_close(encoder.encode(episode), torchhd.multiset(values), atol=0, rtol=0)


def test_preparation_freezes_and_verifies_public_inputs(tmp_path, sample):
    from study import prepare, verify_prepared

    config = replace(sample[1], episodes_per_block=8, budgets=(1,))
    root = tmp_path / "frozen"
    prepare(root, config)
    verify_prepared(root, config)
    path = root / "geography" / "bloor-corridor.geojson"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="geography"):
        verify_prepared(root, config)


def test_config_rejects_empty_and_negative_budgets():
    with pytest.raises(ValueError, match="nonempty"):
        replace(Config(), data_seeds=()).validate()
    with pytest.raises(ValueError, match="positive"):
        replace(Config(), budgets=(-1,)).validate()


def test_learning_update_cost_uses_matched_memory_reviews_and_exact_updates(sample):
    from threadpoolctl import threadpool_limits

    from update_cost import learning_update_cost

    _, config, _, episodes, encoder, raws, _, labels, *_ = sample
    features = torch.stack([explicit_features(e) for e in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    config = replace(
        config, budgets=(2,), update_batches=(1, 4, 40), refit_warmup=1, refit_repeats=2
    )
    settings = {
        "parameters": {
            "2": {
                "logistic_regression": {"C": 1.0, "scaling": "fixed_range"},
                "mlp": {"width": 16, "alpha": 0.1, "scaling": "fixed_range"},
            }
        }
    }
    with threadpool_limits(limits=1):
        result = learning_update_cost(
            encoder, episodes, raws, features, labels, memory, config, settings, 2001
        )
    assert [row["batch_size"] for row in result["skipped_batches"]] == [40]
    for batch in (1, 4):
        rows = [row for row in result["rows"] if row["batch_size"] == batch]
        assert len(rows) == 3
        assert len({tuple(row["initial_review_episode_ids"]) for row in rows}) == 1
        assert len({tuple(row["new_review_episode_ids"]) for row in rows}) == 1
        for row in rows:
            assert row["initial_training_examples"] == 8
            assert row["total_reviewed_examples"] == 8 + batch
            assert not set(row["initial_review_episode_ids"]) & set(row["new_review_episode_ids"])
            assert all(
                e["split"] == "memory"
                for e in episodes
                if e["episode_id"] in row["new_review_episode_ids"]
            )
            timings = row["samples"]
            for i, total in enumerate(timings["complete_wall_ms"]):
                assert total == pytest.approx(
                    sum(timings[key][i] for key in ("encoding_ms", "learning_ms", "prediction_ms"))
                )
            assert row["per_new_sample"]["complete_wall_ms"]["median_ms"] == pytest.approx(
                row["timings"]["complete_wall_ms"]["median_ms"] / batch
            )
        assert next(row for row in rows if row["method"] == "hdc")["additive_memory_rebuild_exact"]
    with pytest.raises(ValueError, match="memory-partition"):
        learning_update_cost(
            encoder,
            episodes,
            raws,
            features,
            labels,
            [i for i, e in enumerate(episodes) if e["split"] == "test"],
            config,
            settings,
            2001,
        )
