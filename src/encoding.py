"""A small algebra: role binding, additive memory, and ordinal permutation."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
import torch
import torchhd
from config import digest

RANGES = {'radio_dbm': (-125., -65.), 'cell_load_pct': (0., 100.),
          'link_loss_pct': (0., 10.), 'link_latency_ms': (0., 120.),
          'peer_radio_dbm': (-125., -65.), 'peer_loss_pct': (0., 10.)}
LOCAL_FIELDS = ('radio_dbm',)
CONTEXT_FIELDS = ('cell_load_pct', 'link_loss_pct', 'link_latency_ms', 'peer_radio_dbm', 'peer_loss_pct')
EXCLUDED = ('subscriber_id', 'phone_id', 'cell_id', 'link_id', 'episode_id', 'world_id',
            'label', 'split', 'decision_s', 'service_disruption', 'longitude', 'latitude')
ENCODER_VERSION = 'telecom-tutorial-v4'


def unit(vector):
    vector = torch.as_tensor(vector, dtype=torch.float32)
    norm = torch.linalg.vector_norm(vector)
    if norm <= 0 or not torch.isfinite(norm):
        raise ValueError('Cannot normalize an empty or non-finite representation')
    return vector / norm


def scaled(value, field):
    low, high = RANGES[field]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Non-finite measurement: {field}')
    return min(1., max(0., (value - low) / (high - low)))


@dataclass(frozen=True)
class Term:
    component: str
    field: str
    value: float | str
    position: int
    weight: float
    factors: tuple[tuple[str, int], ...]
    source_ids: tuple[str, ...]


def factors_for(field):
    if field == 'radio_dbm':
        return (('node:phone', 0), ('attribute:radio', 0))
    path = (('node:phone', 0), ('edge:serves', 1), ('node:cell', 2))
    if field == 'cell_load_pct':
        return path + (('attribute:load', 2),)
    path += (('edge:uses', 3), ('node:backhaul', 4))
    if field.startswith('peer_'):
        path += (('edge:shared_dependency', 5), ('node:peer_phone', 6))
    return path + ((f'attribute:{field}', 6 if field.startswith('peer_') else 4),)


def terms_for(episode, config, path=True, handset=True):
    terms = []
    for position, row in enumerate(episode['observations']):
        terms.append(Term('local', 'radio_dbm', row['radio_dbm'], position,
            config.local_weight / math.sqrt(3), factors_for('radio_dbm'), (row['source_id'],)))
        if path:
            for field in CONTEXT_FIELDS:
                if field == 'cell_load_pct':
                    ids = (row['source_id'], row['cell_state_source_id'])
                elif field.startswith('peer_'):
                    ids = (row['source_id'], row['edge_source_id'], *row['peer_source_ids'], *row.get('peer_edge_ids', []))
                else:
                    ids = (row['source_id'], row['edge_source_id'], row['link_state_source_id'])
                terms.append(Term('context', field, row[field], position,
                    config.context_weight / math.sqrt(15), factors_for(field), tuple(ids)))
        else:
            pool = episode['network_pool'][position]
            count = sum(len(pool[field]) for field in CONTEXT_FIELDS)
            for field in CONTEXT_FIELDS:
                for value, identity in pool[field]:
                    terms.append(Term('context', field, value, position,
                        config.context_weight / math.sqrt(3 * count), ((f'attribute:{field}', 0),), (identity,)))
    if handset and config.handset_weight:
        terms.append(Term('handset', 'handset', episode['handset'], 0,
            config.handset_weight, (('attribute:handset', 0),), (episode['observations'][0]['phone_source_id'],)))
    return terms


class Encoder:
    def __init__(self, config, seed):
        self.config, self.seed = config, seed
        self.dimension = config.dimension
        self.levels = torchhd.level(config.levels, config.dimension, vsa='MAP',
            generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
        self.atom = lru_cache(maxsize=None)(self._atom)
        self.basis = lru_cache(maxsize=None)(self._basis)

    def _atom(self, name):
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}|{name}'.encode()).digest()[:8], 'little') % (2**63 - 1)
        return torchhd.random(1, self.dimension, vsa='MAP',
            generator=torch.Generator().manual_seed(seed), dtype=torch.float32)[0]

    def _basis(self, field, value_index, position, factors, component, binding, sequence):
        value = self.atom(f'value:{value_index}') if field in ('handset', 'categorical') else self.levels[value_index]
        if binding:
            atoms = [torchhd.permute(self.atom(name), shifts=shift) for name, shift in factors]
            value = torchhd.bind(torchhd.multibind(torch.stack(atoms)), value)
        value = torchhd.bind(self.atom(f'channel:{component}'), value)
        return torchhd.permute(value, shifts=position + 1) if sequence and component != 'handset' else value

    def term_vector(self, term, binding=True, sequence=True):
        index = term.value if term.field in ('handset', 'categorical') else round(scaled(term.value, term.field) * (self.config.levels - 1))
        return self.basis(term.field, index, term.position, tuple(tuple(x) for x in term.factors),
            term.component, binding, sequence) * term.weight

    def encode_terms(self, terms, binding=True, sequence=True):
        if not terms:
            return torch.zeros(self.dimension, dtype=torch.float32)
        return torchhd.multiset(torch.stack([self.term_vector(term, binding, sequence) for term in terms])).as_subclass(torch.Tensor)

    def encode(self, episode, path=True, handset=True, binding=True, sequence=True):
        return self.encode_terms(terms_for(episode, self.config, path, handset), binding, sequence)

    def components(self, episode, **kwargs):
        terms = terms_for(episode, self.config, kwargs.get('path', True), kwargs.get('handset', True))
        return {name: self.encode_terms([t for t in terms if t.component == name],
            kwargs.get('binding', True), kwargs.get('sequence', True)) for name in ('local', 'context', 'handset')}

    def manifest(self, episode):
        return {'version': ENCODER_VERSION, 'dimension': self.dimension, 'seed': self.seed,
            'levels': self.config.levels, 'ranges': RANGES, 'config_hash': digest(self.config.to_dict()),
            'normalization': 'L2 only after the float32 additive accumulator', 'excluded_fields': EXCLUDED,
            'terms': [asdict(t) for t in terms_for(episode, self.config)]}

    def signature(self):
        return digest({'version': ENCODER_VERSION, 'seed': self.seed, 'config': self.config.to_dict(),
            'levels_sha256': hashlib.sha256(self.levels.numpy().tobytes()).hexdigest(), 'ranges': RANGES})


def explicit_features(episode, handset=True):
    values = [scaled(row[field], field) for row in episode['observations'] for field in (*LOCAL_FIELDS, *CONTEXT_FIELDS)]
    # A continuous, ordered representation with precisely the same joined paths and numeric inputs.
    if handset:
        values += [float(episode['handset'] == f'model_{i}') * .25 for i in range(3)]
    return torch.tensor(values, dtype=torch.float32)


def structured_scores(query_features, candidate_features):
    return 1 - (candidate_features - query_features).abs().mean(1)
