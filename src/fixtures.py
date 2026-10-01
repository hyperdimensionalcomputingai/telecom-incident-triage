"""Small controlled illustrations, separate from the performance evaluation."""

from copy import deepcopy
from dataclasses import asdict

import pyarrow as pa
import torch
import torchhd

from config import digest
from data import build_episodes
from encoding import Term, unit


def connectivity_pair(tables, episodes, reviews):
    """Rewire one unobserved peer cell's edge; keep every measured value unchanged."""
    selected = None
    for episode in episodes:
        if reviews[episode["episode_id"]]["label"] != "normal":
            continue
        middle = episode["observations"][1]
        peers = [
            r
            for r in tables["peer_observations"].to_pylist()
            if r["timestamp_s"] == middle["timestamp_s"]
        ]
        if middle["cell_id"] not in {r["cell_id"] for r in peers}:
            selected = episode
            break
    if selected is None:
        raise ValueError("No suitable connectivity teaching fixture")
    middle = selected["observations"][1]
    alternate = next(
        r
        for r in tables["links"].to_pylist()
        if r["world_id"] == selected["world_id"] and r["link_id"] != middle["link_id"]
    )
    edges = deepcopy(tables["edges"].to_pylist())
    changed = next(r for r in edges if r["edge_id"] == middle["edge_id"])
    before_edge = deepcopy(changed)
    changed["link_id"] = alternate["link_id"]
    changed.pop("source_sha256")
    changed["source_sha256"] = digest(changed)
    amended = dict(tables, edges=pa.Table.from_pylist(edges, schema=tables["edges"].schema))
    after = next(e for e in build_episodes(amended) if e["episode_id"] == selected["episode_id"])
    return (
        selected,
        after,
        {
            "description": "Counterfactual teaching world: only the serving edge changes; all observations and telemetry stay fixed.",
            "before_edge": before_edge,
            "after_edge": changed,
            "measurements_unchanged": True,
            "evaluation_member": False,
        },
    )


def operator_diagnostics(encoder, tables, episodes, reviews):
    terms_a = [
        Term("concept", "categorical", value, 0, 1.0, ((f"role:{role}", 0),), (f"teaching:{role}",))
        for role, value in (("phone_radio", "good"), ("upstream_link", "bad"))
    ]
    terms_b = [
        Term("concept", "categorical", value, 0, 1.0, ((f"role:{role}", 0),), (f"teaching:{role}",))
        for role, value in (("phone_radio", "bad"), ("upstream_link", "good"))
    ]
    a = encoder.encode_terms(terms_a, sequence=False)
    b = encoder.encode_terms(terms_b, sequence=False)
    bag_a = encoder.encode_terms(terms_a, binding=False, sequence=False)
    bag_b = encoder.encode_terms(terms_b, binding=False, sequence=False)
    role = encoder.atom("role:phone_radio")
    value = encoder.atom("value:good")
    unbound = torchhd.bind(torchhd.bind(role, value), role)
    query = next(e for e in episodes if reviews[e["episode_id"]]["label"] == "radio_deteriorating")
    reversed_episode = deepcopy(query)
    reversed_episode["observations"].reverse()
    graph_before, graph_after, graph_evidence = connectivity_pair(tables, episodes, reviews)
    full = encoder.encode(query)
    handset = encoder.components(query)["handset"]
    rebuilt = encoder.encode(query, handset=False)
    return {
        "role_binding": {
            "bound_cosine": float(unit(a) @ unit(b)),
            "without_binding_max_error": float((bag_a - bag_b).abs().max()),
            "unbinding_exact": bool(torch.equal(unbound, value)),
            "before_terms": [asdict(term) for term in terms_a],
            "after_terms": [asdict(term) for term in terms_b],
        },
        "event_order": {
            "with_order_cosine": float(unit(full) @ unit(encoder.encode(reversed_episode))),
            "without_order_max_error": float(
                (
                    encoder.encode(query, sequence=False)
                    - encoder.encode(reversed_episode, sequence=False)
                )
                .abs()
                .max()
            ),
            "scope": "Explicit counterfactual reordering of a fixed set of events, not an additional measured incident",
        },
        "connectivity": {
            "with_path_cosine": float(
                unit(encoder.encode(graph_before)) @ unit(encoder.encode(graph_after))
            ),
            "without_path_max_error": float(
                (encoder.encode(graph_before, path=False) - encoder.encode(graph_after, path=False))
                .abs()
                .max()
            ),
            "correct_pair_rank": bool(
                float(unit(encoder.encode(graph_before)) @ unit(encoder.encode(graph_before)))
                > float(unit(encoder.encode(graph_before)) @ unit(encoder.encode(graph_after)))
            ),
            "chance_top1": 0.5,
            "candidate_count": 2,
            "evidence": graph_evidence,
            "before_episode": graph_before,
            "after_episode": graph_after,
        },
        "numeric_neighbourhood": {
            "adjacent_cosine": float(unit(encoder.levels[0]) @ unit(encoder.levels[1])),
            "far_cosine": float(unit(encoder.levels[0]) @ unit(encoder.levels[-1])),
        },
        "query_edit": {
            "handset_subtraction_max_error": float((full - handset - rebuilt).abs().max()),
            "scope": "Raw query edit; candidate representations remain unchanged",
        },
        "scope": "Teaching mechanics, not held-out accuracy or a proof of causality",
    }
