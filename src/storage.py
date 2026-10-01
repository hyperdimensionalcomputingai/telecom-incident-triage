"""Exact geography/time selection and exhaustive, prefiltered vector retrieval."""
from pathlib import Path
import json
import lance
import lancedb
import pyarrow as pa
import geodatafusion
from datafusion import SessionContext
import torch
from encoding import unit


def predicate(polygon, start, end):
    # Internal generated WKT, never interpolated free-form user SQL.
    if any(character in polygon for character in ("'", ';')):
        raise ValueError('Unsafe geometry literal')
    return f"timestamp_s >= {int(start)} AND timestamp_s <= {int(end)} AND ST_Intersects(geometry, ST_GeomFromText('{polygon}'))"


class SourceStore:
    def __init__(self, root, tables=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.root))
        if tables is not None:
            crs = str(tables['observations'].schema.field('geometry').type.crs)
            if not any(name in crs for name in ('EPSG:4326', 'OGC:CRS84')):
                raise ValueError('Observation geometry must retain longitude/latitude CRS metadata')
            for name, table in tables.items():
                self.db.create_table(name, data=table, mode='overwrite')
        self.observations = lance.dataset(str(self.root / 'observations.lance'))
        self.episodes = self.db.open_table('episodes').to_arrow().to_pylist()
        self.memory_ids = {r['episode_id'] for r in self.episodes if r['split'] == 'memory'}
        self.ctx = SessionContext()
        geodatafusion.register_all(self.ctx)
        self.ctx.register_record_batches('observations', [self.observations.to_table().to_batches()])

    def eligible(self, polygon, start, end, crosscheck=True):
        where = predicate(polygon, start, end)
        rows = self.observations.to_table(columns=['observation_id', 'episode_id'], filter=where).to_pylist()
        if crosscheck:
            other = self.ctx.sql(f'SELECT observation_id, episode_id FROM observations WHERE {where}').to_arrow_table().to_pylist()
            if sorted(r['observation_id'] for r in rows) != sorted(r['observation_id'] for r in other):
                raise AssertionError('Lance and GeoDataFusion eligibility disagreement')
        groups = {}
        for row in rows:
            groups.setdefault(row['episode_id'], set()).add(row['observation_id'])
        return sorted(identity for identity, observations in groups.items() if len(observations) == 3 and identity in self.memory_ids)


class VectorStore:
    def __init__(self, root, episodes, raws, encoder):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.root))
        metadata = []
        for episode in episodes:
            manifest = encoder.manifest(episode)
            metadata.append({'episode_id': episode['episode_id'], 'world_id': episode['world_id'],
                'split': episode['split'], 'decision_s': episode['decision_s'],
                'manifest_json': json.dumps(manifest, sort_keys=True),
                'source_ids': sorted({identity for term in manifest['terms'] for identity in term['source_ids']})})
        schema = pa.schema([('episode_id', pa.string()), ('world_id', pa.string()), ('split', pa.string()),
                            ('decision_s', pa.int64()), ('manifest_json', pa.string()), ('source_ids', pa.list_(pa.string()))])
        table = pa.Table.from_pylist(metadata, schema=schema)
        normalized = torch.stack([unit(raw) for raw in raws])
        self.quantized = torch.nn.functional.normalize(normalized.to(torch.float16).to(torch.float32), dim=1)
        for name, values, dtype in (('vector', normalized.to(torch.float16), pa.float16()), ('raw_vector', raws, pa.float32())):
            # NumPy is only an Arrow boundary here; all representation math stays in Torch.
            array = pa.array(values.reshape(-1).numpy(), type=dtype)
            table = table.append_column(name, pa.FixedSizeListArray.from_arrays(array, encoder.dimension))
        self.table = self.db.create_table('episodes', data=table, mode='overwrite')

    def search(self, raw, eligible, decision_s, k=10):
        if not eligible:
            return []
        if any("'" in identity for identity in eligible):
            raise ValueError('Unsafe internal episode identity')
        identities = ','.join("'" + identity + "'" for identity in eligible)
        where = f"split = 'memory' AND decision_s < {int(decision_s)} AND episode_id IN ({identities})"
        return (self.table.search(unit(raw).tolist(), vector_column_name='vector').distance_type('cosine')
            .where(where, prefilter=True).bypass_vector_index().select(['episode_id', 'world_id', '_distance'])
            .limit(k).to_arrow().to_pylist())


def directory_bytes(path):
    return sum(file.stat().st_size for file in Path(path).rglob('*') if file.is_file())
