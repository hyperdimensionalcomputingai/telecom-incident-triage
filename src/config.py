"""Frozen study defaults; settings are inputs, never inferred from final outcomes."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('radio_deteriorating', 'transient_recovery', 'shared_transport', 'normal')
BASE_TS = 1788249600  # 2026-09-01 08:00 UTC


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class Config:
    data_seeds: tuple[int, ...] = (1006, 1007, 1008, 1009, 1010)
    encoder_seeds: tuple[int, ...] = (2001, 2002, 2003)
    dimension: int = 4096
    levels: int = 32
    blocks: int = 24
    memory_blocks: int = 12
    validation_blocks: int = 6
    episodes_per_block: int = 100
    local_weight: float = 1.0
    context_weight: float = 2.0
    handset_weight: float = 0.25
    review_delay_s: int = 300
    budgets: tuple[int, ...] = (1, 2, 5, 10, 20)
    bootstrap_repeats: int = 1000
    benchmark_repeats: int = 500
    warmup: int = 50
    refit_repeats: int = 7
    refit_warmup: int = 2
    update_batches: tuple[int, ...] = (1, 20, 40)

    def validate(self):
        if not self.data_seeds or not self.encoder_seeds or not self.budgets:
            raise ValueError('Seeds and learning budgets must be nonempty')
        if min(self.budgets) < 1 or min(self.bootstrap_repeats, self.benchmark_repeats, self.refit_repeats) < 1 or min(self.warmup, self.refit_warmup) < 0:
            raise ValueError('Budgets and measurement counts must be positive; warm-up cannot be negative')
        if self.blocks <= self.memory_blocks + self.validation_blocks:
            raise ValueError('Final testing needs independent network blocks')
        if min(self.memory_blocks, self.validation_blocks) < 1:
            raise ValueError('Memory and validation partitions must be nonempty')
        if self.episodes_per_block % 4 or self.episodes_per_block < 4:
            raise ValueError('Each block needs equal coverage of four patterns')
        if self.dimension < 256 or self.levels < 2:
            raise ValueError('Invalid representation dimensions or numeric levels')
        if self.memory_blocks * self.episodes_per_block // 4 <= max(self.budgets):
            raise ValueError('Insufficient reviewed examples: reserve a further review for the refit measurement')
        if min(self.data_seeds) < 0 or min(self.encoder_seeds) < 0:
            raise ValueError('Seeds must be nonnegative')
        if self.review_delay_s < 0 or min(self.local_weight, self.context_weight, self.handset_weight) < 0:
            raise ValueError('Delays and weights must be nonnegative')
        if not self.update_batches or min(self.update_batches) < 1:
            raise ValueError('Learning-update batches must be positive and nonempty')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        for name in ('data_seeds', 'encoder_seeds', 'budgets', 'update_batches'):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data).validate()
