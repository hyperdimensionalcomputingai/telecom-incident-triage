"""Exact geography/time selection and exhaustive, prefiltered vector retrieval."""

import json
from pathlib import Path

import geodatafusion
import lance
import lancedb
import pyarrow as pa
import torch
from datafusion import SessionContext
from lance_graph import CypherQuery, GraphConfig

from encoding import unit

# A property-graph view over the stored source tables; no graph copy is written.
# Each relationship reuses a table that already holds both endpoint IDs.
NODES = {
    "Episode": ("episodes", "episode_id"),
    "Observation": ("observations", "observation_id"),
    "Phone": ("phones", "phone_id"),
    "Cell": ("cells", "cell_id"),
    "Link": ("links", "link_id"),
    "CellStatus": ("cell_status", "source_id"),
    "LinkStatus": ("link_status", "source_id"),
    "Peer": ("peer_observations", "peer_observation_id"),
}
RELATIONSHIPS = {
    "HAS_OBSERVATION": ("observations", "episode_id", "observation_id"),
    "FROM_PHONE": ("observations", "observation_id", "phone_id"),
    "SERVED_BY": ("observations", "observation_id", "cell_id"),
    "BACKHAULED_BY": ("edges", "cell_id", "link_id"),
    "CELL_STATE": ("cell_status", "cell_id", "source_id"),
    "LINK_STATE": ("link_status", "link_id", "source_id"),
    "HOSTS_PEER": ("peer_observations", "cell_id", "peer_observation_id"),
}
builder = GraphConfig.builder()
for label, (_, identity) in NODES.items():
    builder = builder.with_node_label(label, identity)
for kind, (_, source, target) in RELATIONSHIPS.items():
    builder = builder.with_relationship(kind, source, target)
GRAPH = builder.build()
# Phone -> serving cell -> backhaul link valid at that time, with same-time cell
# and link telemetry. Mirrors build_episodes, but as one declarative path.
PATH = CypherQuery(
    """
    MATCH (e:Episode)-[:HAS_OBSERVATION]->(o:Observation)-[:SERVED_BY]->(c:Cell)-[b:BACKHAULED_BY]->(l:Link),
          (o)-[:FROM_PHONE]->(ph:Phone),
          (c)-[:CELL_STATE]->(cs:CellStatus),
          (l)-[:LINK_STATE]->(ls:LinkStatus)
    WHERE e.episode_id = $episode
      AND o.timestamp_s >= b.valid_from_s AND o.timestamp_s < b.valid_to_s
      AND cs.timestamp_s = o.timestamp_s AND ls.timestamp_s = o.timestamp_s
    RETURN o.ordinal, o.observation_id, o.radio_dbm, ph.phone_id, ph.handset, c.cell_id,
           b.edge_id, l.link_id, cs.source_id, cs.cell_load_pct, ls.source_id,
           ls.link_loss_pct, ls.link_latency_ms
    ORDER BY o.ordinal
    """
).with_config(GRAPH)
# Other phones whose cells share that link at the same time, via their own edge.
PEERS = CypherQuery(
    """
    MATCH (e:Episode)-[:HAS_OBSERVATION]->(o:Observation)-[:SERVED_BY]->(c:Cell)-[b:BACKHAULED_BY]->(l:Link)<-[pb:BACKHAULED_BY]-(:Cell)-[:HOSTS_PEER]->(p:Peer)
    WHERE e.episode_id = $episode
      AND o.timestamp_s >= b.valid_from_s AND o.timestamp_s < b.valid_to_s
      AND p.timestamp_s = o.timestamp_s AND p.edge_id = pb.edge_id
    RETURN o.ordinal, l.link_id, p.peer_observation_id, p.edge_id, p.peer_radio_dbm,
           p.peer_loss_pct
    ORDER BY o.ordinal, p.peer_observation_id
    """
).with_config(GRAPH)


def predicate(polygon, start, end):
    # Internal generated WKT, never interpolated free-form user SQL.
    if any(character in polygon for character in ("'", ";")):
        raise ValueError("Unsafe geometry literal")
    return f"timestamp_s >= {int(start)} AND timestamp_s <= {int(end)} AND ST_Intersects(geometry, ST_GeomFromText('{polygon}'))"


class SourceStore:
    def __init__(self, root, tables=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.root))
        if tables is not None:
            crs = str(tables["observations"].schema.field("geometry").type.crs)
            if not any(name in crs for name in ("EPSG:4326", "OGC:CRS84")):
                raise ValueError("Observation geometry must retain longitude/latitude CRS metadata")
            for name, table in tables.items():
                self.db.create_table(name, data=table, mode="overwrite")
        self.observations = lance.dataset(str(self.root / "observations.lance"))
        self.episodes = self.db.open_table("episodes").to_arrow().to_pylist()
        self.memory_ids = {r["episode_id"] for r in self.episodes if r["split"] == "memory"}
        self.ctx = SessionContext()
        geodatafusion.register_all(self.ctx)
        self.ctx.register_record_batches(
            "observations", [self.observations.to_table().to_batches()]
        )
        # Read each Lance table once; names sharing a table share one Arrow table.
        # shortcut: whole tables in memory, fine at tutorial scale; upgrade to lazy
        # Lance providers once lance-graph's CypherEngine supports them with parameters.
        arrow = {
            name: self.db.open_table(name).to_arrow()
            for name in {spec[0] for spec in (*NODES.values(), *RELATIONSHIPS.values())}
        }
        self.graph = {name: arrow[spec[0]] for name, spec in {**NODES, **RELATIONSHIPS}.items()}

    def eligible(self, polygon, start, end, crosscheck=True):
        where = predicate(polygon, start, end)
        rows = self.observations.to_table(
            columns=["observation_id", "episode_id"], filter=where
        ).to_pylist()
        if crosscheck:
            other = (
                self.ctx.sql(f"SELECT observation_id, episode_id FROM observations WHERE {where}")
                .to_arrow_table()
                .to_pylist()
            )
            if sorted(r["observation_id"] for r in rows) != sorted(
                r["observation_id"] for r in other
            ):
                raise AssertionError("Lance and GeoDataFusion eligibility disagreement")
        groups = {}
        for row in rows:
            groups.setdefault(row["episode_id"], set()).add(row["observation_id"])
        # Eligibility requires all three episode observations inside the requested
        # map/time region. This is an exact record filter, not hypervector similarity.
        return sorted(
            identity
            for identity, observations in groups.items()
            if len(observations) == 3 and identity in self.memory_ids
        )

    def evidence_trail(self, episode_id):
        """Follow a retrieved episode's graph path to the records an analyst can inspect."""
        path, peers = (
            query.with_parameter("episode", episode_id).execute(self.graph).to_pylist()
            for query in (PATH, PEERS)
        )
        keys = ("ph.phone_id", "o.observation_id", "b.edge_id", "cs.source_id", "ls.source_id")
        return {
            "path": path,
            "peers": peers,
            "source_ids": sorted(
                {row[key] for row in path for key in keys}
                | {row[key] for row in peers for key in ("p.peer_observation_id", "p.edge_id")}
            ),
        }


class VectorStore:
    def __init__(self, root, episodes, raws, encoder):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.root))
        metadata = []
        for episode in episodes:
            manifest = encoder.manifest(episode)
            metadata.append(
                {
                    "episode_id": episode["episode_id"],
                    "world_id": episode["world_id"],
                    "split": episode["split"],
                    "decision_s": episode["decision_s"],
                    "manifest_json": json.dumps(manifest, sort_keys=True),
                    "source_ids": sorted(
                        {identity for term in manifest["terms"] for identity in term["source_ids"]}
                    ),
                }
            )
        schema = pa.schema(
            [
                ("episode_id", pa.string()),
                ("world_id", pa.string()),
                ("split", pa.string()),
                ("decision_s", pa.int64()),
                ("manifest_json", pa.string()),
                ("source_ids", pa.list_(pa.string())),
            ]
        )
        table = pa.Table.from_pylist(metadata, schema=schema)
        # Normalize only after all weighted episode terms are bundled. Keep raw
        # float32 sums as well, so component removal and reconstruction are possible.
        normalized = torch.stack([unit(raw) for raw in raws])
        # float16 storage rounds coordinates and slightly changes vector length.
        # Re-normalize the rounded values in float32 for the in-memory cosine
        # crosscheck; this does not recover the precision lost during storage.
        self.quantized = torch.nn.functional.normalize(
            normalized.to(torch.float16).to(torch.float32), dim=1
        )
        for name, values, dtype in (
            ("vector", normalized.to(torch.float16), pa.float16()),
            ("raw_vector", raws, pa.float32()),
        ):
            # NumPy is only an Arrow boundary here; all representation math stays in Torch.
            array = pa.array(values.reshape(-1).numpy(), type=dtype)
            table = table.append_column(
                name, pa.FixedSizeListArray.from_arrays(array, encoder.dimension)
            )
        self.table = self.db.create_table("episodes", data=table, mode="overwrite")

    def search(self, raw, eligible, decision_s, k=10):
        if not eligible:
            return []
        if any("'" in identity for identity in eligible):
            raise ValueError("Unsafe internal episode identity")
        identities = ",".join("'" + identity + "'" for identity in eligible)
        where = (
            f"split = 'memory' AND decision_s < {int(decision_s)} AND episode_id IN ({identities})"
        )
        # Compare directions using cosine distance (1 - cosine similarity), with
        # exact eligibility applied first. Exhaustive search avoids approximations
        # from a vector index; float16 rounding remains a separate approximation.
        return (
            self.table.search(unit(raw).tolist(), vector_column_name="vector")
            .distance_type("cosine")
            .where(where, prefilter=True)
            .bypass_vector_index()
            .select(["episode_id", "world_id", "_distance"])
            .limit(k)
            .to_arrow()
            .to_pylist()
        )


def directory_bytes(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())
