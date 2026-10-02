"""Independent evidence-based labels; no generator scenario flag or encoder features."""

from config import LABELS, digest


def derive_reviews(rows, delay_s):
    observations = {r["observation_id"]: r for r in rows["observations"]}
    edges = {}
    for edge in rows["edges"]:
        edges.setdefault(edge["cell_id"], []).append(edge)
    link_status = {(r["link_id"], r["timestamp_s"]): r for r in rows["link_status"]}
    peer_groups = {}
    for peer in rows["peer_observations"]:
        choices = [
            e
            for e in edges[peer["cell_id"]]
            if e["valid_from_s"] <= peer["timestamp_s"] < e["valid_to_s"]
        ]
        if len(choices) != 1:
            raise ValueError("Ambiguous or absent peer topology")
        peer_groups.setdefault((choices[0]["link_id"], peer["timestamp_s"]), []).append(peer)
    reviews = []
    for episode in rows["episodes"]:
        obs = sorted(
            (observations[identity] for identity in episode["observation_ids"]),
            key=lambda r: r["timestamp_s"],
        )
        witness = [r["source_id"] for r in obs]
        shared = False
        for observation in obs:
            choices = [
                e
                for e in edges[observation["cell_id"]]
                if e["valid_from_s"] <= observation["timestamp_s"] < e["valid_to_s"]
            ]
            if len(choices) != 1:
                raise ValueError("Ambiguous or absent serving topology")
            edge = choices[0]
            state = link_status[edge["link_id"], observation["timestamp_s"]]
            peers = peer_groups[edge["link_id"], observation["timestamp_s"]]
            # Synthetic evidence rule: >=3% link loss plus >=3% loss for every
            # peer, with at least two peers, marks a shared transport pattern.
            # The generator's healthy loss tops out at 1.8%, while stressed loss
            # starts at 3.2%; 3% sits in that deliberately constructed gap.
            # These thresholds define this demonstration, not a carrier diagnostic
            # standard. Matching the rule does not establish the actual root cause.
            shared |= (
                state["link_loss_pct"] >= 3
                and len(peers) >= 2
                and all(p["peer_loss_pct"] >= 3 for p in peers)
            )
            witness.extend(
                [edge["source_id"], state["source_id"], *[p["source_id"] for p in peers]]
            )
        # End minus start in dBm: negative means weakening, positive means recovery.
        # A 15 dB change is our chosen pattern threshold. Shared impairment takes
        # precedence because it can coexist with either phone signal trajectory.
        delta = obs[-1]["radio_dbm"] - obs[0]["radio_dbm"]
        label = (
            LABELS[2]
            if shared
            else LABELS[0]
            if delta <= -15
            else LABELS[1]
            if delta >= 15
            else LABELS[3]
        )
        identity = episode["episode_id"] + "-REVIEW"
        record = {
            "source_id": identity,
            "episode_id": episode["episode_id"],
            "world_id": episode["world_id"],
            "label": label,
            "label_status": "simulated_evidence_rule",
            "label_is_root_cause": False,
            # Simulated feedback appears after the observation window; this delay
            # never changes the facts available to the representation at decision.
            "ready_s": episode["decision_s"] + delay_s,
            "witness_source_ids": sorted(set(witness)),
        }
        record["source_sha256"] = digest(record)
        reviews.append(record)
    return reviews
