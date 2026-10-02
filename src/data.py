"""Consistent synthetic network worlds on a frozen public street corridor."""

from pathlib import Path

import geoarrow.pyarrow as ga
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from config import BASE_TS, LABELS, digest, save_json
from geography import load_geography, route_points

TABLES = (
    "subscribers",
    "phones",
    "cells",
    "links",
    "edges",
    "cell_status",
    "link_status",
    "observations",
    "peer_observations",
    "episodes",
    "reviews",
)


def source(record, identity):
    row = dict(record, source_id=identity)
    row["source_sha256"] = digest(row)
    return row


def generate(root, config, seed, geography_root=None):
    """Generate controlled patterns, not a physical mobile-network simulation.

    A subscriber is a customer, a cell supplies a phone's wireless connection,
    and a backhaul link carries that cell's traffic onward. Six cells share two
    links per independent network/time block. All telemetry (component-state
    measurements) is synthetic; public road geometry does not determine signal.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lines, geography_manifest = (
        load_geography(geography_root) if geography_root else load_geography()
    )
    rng = np.random.default_rng(seed)
    rows = {name: [] for name in TABLES}
    for block in range(config.blocks):
        prefix = f"S{seed}-W{block:02d}"
        partition = (
            "memory"
            if block < config.memory_blocks
            else "validation"
            if block < config.memory_blocks + config.validation_blocks
            else "test"
        )
        # Separate blocks by one day (86,400 seconds) for a simple ordered timeline.
        beginning = BASE_TS + block * 86400
        # Each slot supplies one episode of each of the four patterns. Balanced
        # counts make this a controlled experiment, not realistic incident prevalence.
        slots = config.episodes_per_block // len(LABELS)
        middle = slots // 2
        # Six cells / two links gives three cells per shared dependency: enough
        # for a three-observation serving-cell route and peers on other cells.
        # This is a small illustrative topology, not a model of a carrier network.
        cells = [f"{prefix}-C{i}" for i in range(6)]
        links = [f"{prefix}-L{i}" for i in range(2)]
        for cell in cells:
            rows["cells"].append(
                source({"cell_id": cell, "world_id": prefix, "technology": "NR"}, cell)
            )
        for link in links:
            rows["links"].append(
                source({"link_id": link, "world_id": prefix, "medium": "fibre"}, link)
            )
        mappings = []
        for phase in range(2):
            # Randomly assign three cells to each link in each phase. Rewiring
            # halfway through the slot sequence makes same-time connectivity matter.
            order = rng.permutation(6)
            mapping = {cells[int(c)]: links[i // 3] for i, c in enumerate(order)}
            mappings.append(mapping)
            start = beginning if phase == 0 else beginning + middle * 120
            end = beginning + middle * 120 if phase == 0 else beginning + 86400
            for i, cell in enumerate(cells):
                edge_id = f"{prefix}-EDGE{phase}-{i}"
                rows["edges"].append(
                    source(
                        {
                            "edge_id": edge_id,
                            "world_id": prefix,
                            "cell_id": cell,
                            "link_id": mapping[cell],
                            "valid_from_s": start,
                            "valid_to_s": end,
                            "available_s": start,
                        },
                        edge_id,
                    )
                )
        for slot in range(slots):
            phase = int(slot >= middle)
            mapping = mappings[phase]
            # Two-minute slots leave a gap between the 40-second observation windows.
            start = beginning + slot * 120
            impaired = links[int(rng.integers(2))]
            healthy = next(link for link in links if link != impaired)
            # An episode is a 40-second window with three samples, 20 seconds apart.
            # The encoder tags their ordinal positions, not elapsed seconds.
            times = [start + pos * 20 for pos in range(3)]
            loss_by_link = {}
            for pos, timestamp in enumerate(times):
                for i, cell in enumerate(cells):
                    # Loads vary independently between 25% and 90%: contextual
                    # background variation, not the rule that defines any label.
                    identity = f"{prefix}-CS{slot}-{pos}-{i}"
                    rows["cell_status"].append(
                        source(
                            {
                                "cell_id": cell,
                                "timestamp_s": timestamp,
                                "available_s": timestamp,
                                "cell_load_pct": float(rng.uniform(25, 90)),
                            },
                            identity,
                        )
                    )
                for i, link in enumerate(links):
                    # Packet loss is a percentage of missing pieces of transmitted
                    # data; latency is delay in milliseconds. Only one link's middle
                    # sample is stressed. These sampled ranges deliberately create
                    # separable patterns; they are not fitted carrier distributions.
                    # Healthy loss is <=1.8%, stressed loss >=3.2%, leaving a gap
                    # around the review rule's 3% threshold. Delay ranges overlap,
                    # so high delay alone does not define shared impairment.
                    stressed = link == impaired and pos == 1
                    loss = float(rng.uniform(3.2, 8.0) if stressed else rng.uniform(0.05, 1.8))
                    latency = float(rng.uniform(28, 100) if stressed else rng.uniform(8, 48))
                    loss_by_link[link, pos] = loss
                    identity = f"{prefix}-LS{slot}-{pos}-{i}"
                    rows["link_status"].append(
                        source(
                            {
                                "link_id": link,
                                "timestamp_s": timestamp,
                                "available_s": timestamp,
                                "link_loss_pct": loss,
                                "link_latency_ms": latency,
                            },
                            identity,
                        )
                    )
                    attached = [cell for cell in cells if mapping[cell] == link]
                    # Two peers supply corroborating shared-link evidence and a
                    # simple average. The review rule also requires at least two.
                    for peer in range(2):
                        # Each peer record represents another phone using this link.
                        # Its loss copies the shared link's loss with small noise:
                        # correlated trouble is built into the synthetic scenario.
                        # Peer identities are not separately encoded; build_episodes
                        # reduces the two records to two mean measurements per time.
                        identity = f"{prefix}-PEER{slot}-{pos}-{i}-{peer}"
                        cell = attached[peer]
                        edge_id = f"{prefix}-EDGE{phase}-{cells.index(cell)}"
                        rows["peer_observations"].append(
                            source(
                                {
                                    "peer_observation_id": identity,
                                    "world_id": prefix,
                                    "cell_id": cell,
                                    "edge_id": edge_id,
                                    "timestamp_s": timestamp,
                                    "available_s": timestamp,
                                    "peer_radio_dbm": float(rng.uniform(-94, -74)),
                                    "peer_loss_pct": loss + float(rng.uniform(-0.02, 0.02)),
                                },
                                identity,
                            )
                        )
            # Four contemporaneous subscribers see the same network, not four incompatible snapshots.
            for incident_number, category in enumerate(rng.permutation(4)):
                category = int(category)
                episode_id = f"{prefix}-E{slot:02d}-{incident_number}"
                phone_id = f"{prefix}-PHONE{slot:02d}-{incident_number}"
                subscriber_id = f"{prefix}-PERSON{slot:02d}-{incident_number}"
                # Three arbitrary phone categories, sampled independently of the
                # pattern: handset model is deliberately an incidental feature.
                handset = f"model_{int(rng.integers(3))}"
                rows["subscribers"].append(
                    source(
                        {
                            "subscriber_id": subscriber_id,
                            "world_id": prefix,
                            "pseudonym": f"Traveller {block}-{slot}-{category}",
                        },
                        subscriber_id,
                    )
                )
                rows["phones"].append(
                    source(
                        {
                            "phone_id": phone_id,
                            "subscriber_id": subscriber_id,
                            "world_id": prefix,
                            "handset": handset,
                        },
                        phone_id,
                    )
                )
                link = impaired if category == 2 else healthy
                attached = [cell for cell in cells if mapping[cell] == link]
                route = [attached[int(i)] for i in rng.permutation(3)]
                # A shared link problem can accompany weakening, recovering or
                # steady phone signal. This is why signal alone cannot identify it.
                profile = category if category != 2 else int(rng.choice([0, 1, 3]))
                # dBm is logarithmic signal power; more negative means weaker.
                # These three hand-designed trajectories are independent of map
                # position; they are not predictions of wireless propagation.
                # Their endpoints differ by at least 21 dB for weakening/recovery,
                # leaving a margin around the review rule's +/-15 dB thresholds.
                # Steady signal varies at most 6 dB across the +/-3 dB jitter.
                if profile == 0:
                    radio = [rng.uniform(-84, -76), rng.uniform(-102, -91), rng.uniform(-116, -108)]
                elif profile == 1:
                    radio = [rng.uniform(-114, -106), rng.uniform(-101, -91), rng.uniform(-85, -75)]
                else:
                    centre = rng.uniform(-96, -80)
                    radio = [centre + rng.uniform(-3, 3) for _ in range(3)]
                points = route_points(lines, rng)
                observation_ids = []
                for pos, (cell, timestamp, point) in enumerate(zip(route, times, points)):
                    identity = f"{episode_id}-O{pos}"
                    observation_ids.append(identity)
                    edge_id = f"{prefix}-EDGE{phase}-{cells.index(cell)}"
                    rows["observations"].append(
                        source(
                            {
                                "observation_id": identity,
                                "episode_id": episode_id,
                                "world_id": prefix,
                                "phone_id": phone_id,
                                "cell_id": cell,
                                "edge_id": edge_id,
                                "ordinal": pos,
                                "timestamp_s": timestamp,
                                "available_s": timestamp,
                                "longitude": point[0],
                                "latitude": point[1],
                                "radio_dbm": float(radio[pos]),
                                "service_disruption": category != 3 and pos == 1,
                            },
                            identity,
                        )
                    )
                rows["episodes"].append(
                    source(
                        {
                            "episode_id": episode_id,
                            "world_id": prefix,
                            "phone_id": phone_id,
                            "subscriber_id": subscriber_id,
                            "split": partition,
                            "decision_s": times[-1],
                            "observation_ids": observation_ids,
                        },
                        episode_id,
                    )
                )
    # Labels are independently reconstructed from records and edges; category isn't written to features.
    from oracle import derive_reviews

    rows["reviews"] = derive_reviews(rows, config.review_delay_s)
    tables = {}
    for name, records in rows.items():
        table = pa.Table.from_pylist(records)
        if name == "observations":
            geometry = ga.as_wkb(
                ga.make_point(
                    [r["longitude"] for r in records],
                    [r["latitude"] for r in records],
                    crs="EPSG:4326",
                )
            )
            table = table.append_column("geometry", geometry)
        pq.write_table(table, root / f"{name}.parquet")
        tables[name] = table
    counts = {name: len(records) for name, records in rows.items()}
    manifest = {
        "data_seed": seed,
        "counts": counts,
        "geography": geography_manifest,
        "config_hash": digest(config.to_dict()),
        "data_hash": digest(rows),
        "labels": "Independent rule checker over synthetic source evidence; no human review occurred",
        "design": "Balanced controlled patterns; neither real prevalence nor a radio propagation model",
        "split_unit": "Entire network world and timeline; disjoint subscriber and incident identities",
    }
    save_json(root / "manifest.json", manifest)
    return tables


def load_tables(root):
    return {name: pq.read_table(Path(root) / f"{name}.parquet") for name in TABLES}


def source_lookup(tables):
    result = {}
    for table in tables.values():
        for row in table.to_pylist():
            row.pop("geometry", None)
            expected = row.pop("source_sha256")
            if digest(row) != expected:
                raise ValueError(f"Source checksum mismatch: {row['source_id']}")
            row["source_sha256"] = expected
            if row["source_id"] in result:
                raise ValueError("Duplicate source identity")
            result[row["source_id"]] = row
    return result


def build_episodes(tables):
    """Resolve connected facts before turning any measurements into hypervectors.

    For each observation: phone -> serving cell -> time-valid backhaul link, then
    that cell's load, that link's loss/delay, and peers sharing that link at that
    exact time. IDs select the evidence but are not themselves encoded features.
    Missing/duplicate/stale records fail instead of inventing measurements.
    """
    lookup = source_lookup(tables)

    def frame(name):
        return pl.from_arrow(
            tables[name].drop(["geometry"])
            if "geometry" in tables[name].column_names
            else tables[name]
        )

    def rename_source(df, prefix):
        return df.rename(
            {"source_id": f"{prefix}_source_id", "source_sha256": f"{prefix}_source_sha256"}
        )

    phones = rename_source(frame("phones"), "phone").drop("world_id")
    edges = rename_source(frame("edges"), "edge").drop("world_id")
    cell_state = rename_source(frame("cell_status"), "cell_state").rename(
        {"available_s": "cell_available_s"}
    )
    link_state = rename_source(frame("link_status"), "link_state").rename(
        {"available_s": "link_available_s"}
    )
    peer = frame("peer_observations").join(
        edges.select("edge_id", "cell_id", "link_id", "valid_from_s", "valid_to_s"),
        on=["edge_id", "cell_id"],
        how="left",
        validate="m:1",
    )
    if peer["link_id"].null_count():
        raise ValueError("Missing peer topology edge")
    # Validity intervals are half-open: start <= time < end. At a rewire boundary,
    # the old edge has expired and the new edge applies, avoiding two active edges.
    if peer.filter(
        (pl.col("timestamp_s") < pl.col("valid_from_s"))
        | (pl.col("timestamp_s") >= pl.col("valid_to_s"))
    ).height:
        raise ValueError("Peer observation uses a stale topology edge")
    # Average only peers on the same link at the same time, with equal weight.
    # Averaging dBm operates in logarithmic units, not mean physical power in watts.
    # Each average becomes one encoded term; individual variation and peer count
    # are lost to this representation, while source IDs remain for inspection.
    summary = peer.group_by(["link_id", "timestamp_s"]).agg(
        pl.col("peer_radio_dbm").mean(),
        pl.col("peer_loss_pct").mean(),
        pl.col("source_id").alias("peer_source_ids"),
        pl.col("edge_id").alias("peer_edge_ids"),
        pl.col("available_s").max().alias("peer_available_s"),
    )
    # Exact-time joins assume complete, synchronized snapshots. There is no nearest
    # timestamp lookup, interpolation, missing-value imputation, or learned join.
    # m:1 requires one matching infrastructure record per phone observation.
    df = (
        frame("observations")
        .join(phones, on="phone_id", how="left", validate="m:1")
        .join(
            edges.rename({"available_s": "edge_available_s"}),
            on=["edge_id", "cell_id"],
            how="left",
            validate="m:1",
        )
        .join(cell_state, on=["cell_id", "timestamp_s"], how="left", validate="m:1")
        .join(link_state, on=["link_id", "timestamp_s"], how="left", validate="m:1")
        .join(summary, on=["link_id", "timestamp_s"], how="left", validate="m:1")
        .sort(["episode_id", "ordinal"])
    )
    required = [
        "handset",
        "link_id",
        "valid_from_s",
        "cell_load_pct",
        "link_loss_pct",
        "link_latency_ms",
        "peer_radio_dbm",
        "peer_loss_pct",
    ]
    if df.height != tables["observations"].num_rows or any(df[c].null_count() for c in required):
        raise ValueError("Missing or nonunique factual joins")
    if df.filter(
        (pl.col("timestamp_s") < pl.col("valid_from_s"))
        | (pl.col("timestamp_s") >= pl.col("valid_to_s"))
    ).height:
        raise ValueError("Observation uses a stale topology edge")
    states = {}
    for name in ("cell_status", "link_status", "peer_observations"):
        for row in tables[name].to_pylist():
            states.setdefault(row["timestamp_s"], []).append(row["source_id"])
    grouped = {
        key[0]: group.to_dicts() for key, group in df.group_by("episode_id", maintain_order=True)
    }
    episodes = []
    for meta in tables["episodes"].to_pylist():
        observations = grouped.get(meta["episode_id"], [])
        # The encoder's sqrt(3)/sqrt(15) weights rely on this fixed episode shape.
        if len(observations) != 3 or [o["ordinal"] for o in observations] != [0, 1, 2]:
            raise ValueError("An episode requires three unique ordered observations")
        if sorted(meta["observation_ids"]) != sorted(o["observation_id"] for o in observations):
            raise ValueError("Episode observation identities disagree with source metadata")
        for observation in observations:
            # Measurement time and availability time can differ. Every joined
            # source must already be available at the decision (third observation),
            # so later information cannot improve an earlier representation.
            if (
                max(
                    observation[key]
                    for key in (
                        "available_s",
                        "edge_available_s",
                        "cell_available_s",
                        "link_available_s",
                        "peer_available_s",
                    )
                )
                > meta["decision_s"]
            ):
                raise ValueError("Features contain future information")
        if meta["decision_s"] != max(o["timestamp_s"] for o in observations):
            raise ValueError("Decision timestamp does not match the available episode")
        # The no-path comparison uses all same-time component/peer measurements,
        # regardless of connection to this phone. Keep that pool separate from the
        # five selected context fields used by the default representation.
        pool = []
        for observation in observations:
            values = {
                field: []
                for field in (
                    "cell_load_pct",
                    "link_loss_pct",
                    "link_latency_ms",
                    "peer_radio_dbm",
                    "peer_loss_pct",
                )
            }
            for identity in states[observation["timestamp_s"]]:
                record = lookup[identity]
                if record["available_s"] > meta["decision_s"]:
                    raise ValueError("Network context contains future information")
                for field, measurements in values.items():
                    if field in record:
                        measurements.append((record[field], identity))
            pool.append(values)
        episodes.append(
            dict(
                meta,
                handset=observations[0]["handset"],
                observations=observations,
                network_pool=pool,
                network_source_ids=sorted(
                    {identity for o in observations for identity in states[o["timestamp_s"]]}
                ),
            )
        )
    return episodes
