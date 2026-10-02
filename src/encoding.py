"""Represent one episode: phone signal + connected network context + handset model.

An episode has three observation times. Each supplies one phone signal measurement
and five connected measurements: cell load, link loss/delay, and two peer averages.
Together with one handset category, these become 3 + 15 + 1 weighted terms in one
hypervector. MAP binding multiplies coordinates, permutation rotates them, and
bundling adds them without thresholding. These are representation operations;
the separately implemented prototype learner consumes their result later.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch
import torchhd

from config import digest

# Chosen demo bounds, fixed before encoding; these are not observed dataset minima
# and maxima, universal telecom limits, or a calibration of connection quality.
# dBm measures signal power logarithmically (less negative is stronger), percentages
# measure load/loss, and milliseconds measure delay. Linear scaling is performed
# in these stated units: equal dBm steps are not equal steps in physical watts.
# Compare these bounds with generate() in data.py: phone signal spans roughly
# [-116, -75] dBm and peer signal [-94, -74]. The common [-125, -65] interval
# covers both with spare room. This is a practical way to understand the bounds;
# the exact endpoints are modelling choices. With 32 levels, the spacing between
# adjacent level centres is 60/31 = 1.94 dB.
# Wider bounds give coarser resolution; narrower bounds clip more measurements.
RANGES = {
    "radio_dbm": (-125.0, -65.0),
    # Load uses the full percentage scale; generated loads are only 25-90%.
    "cell_load_pct": (0.0, 100.0),
    # Loss is physically a 0-100% fraction, but this demo focuses its levels on
    # 0-10%: generated link loss stays below 8%, so 10 is a chosen encoding cap.
    "link_loss_pct": (0.0, 10.0),
    # Generated delays are 8-100 ms. Zero is the nonnegative origin, while 120
    # leaves headroom; that exact upper cap is a heuristic, not a service standard.
    "link_latency_ms": (0.0, 120.0),
    "peer_radio_dbm": (-125.0, -65.0),
    # Peer loss tracks link loss with +/-0.02 percentage points of noise, staying
    # below 8.02%; reuse the same chosen 0-10% scale as link loss.
    "peer_loss_pct": (0.0, 10.0),
}
# The existing internal name "local" means only "phone signal" in this demo.
# The asymmetry is a feature choice: one phone measurement versus five connected
# measurements per time. HDC does not require this division or these term counts.
LOCAL_FIELDS = ("radio_dbm",)
CONTEXT_FIELDS = (
    "cell_load_pct",
    "link_loss_pct",
    "link_latency_ms",
    "peer_radio_dbm",
    "peer_loss_pct",
)
# This tuple documents the manifest; it does not itself filter input dictionaries.
# terms_for() explicitly selects the encoded fields. IDs support joins/provenance,
# location supports exact filtering, and labels/outcomes must not become features.
EXCLUDED = (
    "subscriber_id",
    "phone_id",
    "cell_id",
    "link_id",
    "episode_id",
    "world_id",
    "label",
    "split",
    "decision_s",
    "service_disruption",
    "longitude",
    "latitude",
)
ENCODER_VERSION = "telecom-tutorial-v4"


def unit(vector):
    """Return z / ||z||_2, preserving direction and giving unit Euclidean length.

    The norm is sqrt(sum(z_i**2)). Unit-vector dot products give cosine similarity.
    Normalize the completed bundle, not each term/component separately: doing so
    earlier would change their relative weights. Keep the raw bundle for edits.
    """
    vector = torch.as_tensor(vector, dtype=torch.float32)
    norm = torch.linalg.vector_norm(vector)
    if norm <= 0 or not torch.isfinite(norm):
        raise ValueError("Cannot normalize an empty or non-finite representation")
    return vector / norm


def scaled(value, field):
    """Locate a measurement in its fixed range using inverse linear interpolation.

    Forward interpolation is x = low + s * (high - low). Solving for its fraction
    gives s = (x - low) / (high - low): the numerator is distance from the lower
    bound; the denominator is the whole interval width. This ordinary affine
    scaling assumes equal steps in the field's units deserve equal scale steps.
    For radio -80 dBm in [-125, -65], s = 45 / 60 = 0.75, not 75% signal quality.
    Clipping makes values beyond a bound indistinguishable from that endpoint.
    """
    low, high = RANGES[field]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite measurement: {field}")
    return min(1.0, max(0.0, (value - low) / (high - low)))


@dataclass(frozen=True)
class Term:
    """One measurement's recipe, retaining its source evidence outside the vector.

    position is observation order (0/1/2). Each (name, shift) in factors describes
    a typed role's structural path position; it is a different use of permutation.
    weight is applied after binding/rotation. source_ids never enter the arithmetic.
    """

    component: str
    field: str
    value: float | str
    position: int
    weight: float
    factors: tuple[tuple[str, int], ...]
    source_ids: tuple[str, ...]


def factors_for(field):
    """Describe what was measured and where it sits relative to our phone.

    A serving cell provides the phone's wireless connection; its backhaul carries
    traffic onward. Peers are other phones sharing that backhaul. Structural shifts
    tag phone=0, serving relation=1, cell=2, uses relation=3, backhaul=4, sharing=5,
    peer=6. They are schema choices, not distances, elapsed times, or record IDs.
    An attribute gets its object's shift: cell load=2, link measurements=4, peers=6.
    Distinct atom names still distinguish an object from its attribute at that shift.
    """
    if field == "radio_dbm":
        return (("node:phone", 0), ("attribute:radio", 0))
    path = (("node:phone", 0), ("edge:serves", 1), ("node:cell", 2))
    if field == "cell_load_pct":
        return path + (("attribute:load", 2),)
    path += (("edge:uses", 3), ("node:backhaul", 4))
    if field.startswith("peer_"):
        path += (("edge:shared_dependency", 5), ("node:peer_phone", 6))
    return path + ((f"attribute:{field}", 6 if field.startswith("peer_") else 4),)


def terms_for(episode, config, path=True, handset=True):
    """Create 19 terms for the default, already joined three-observation episode.

    For N independent zero-mean bipolar terms of dimension D, the expected squared
    norm of their sum is N*D. Dividing each by sqrt(N) compensates for term count
    before applying the component weight. Actual terms can be correlated, so this
    is approximate balancing, not exact norm equality or a fixed similarity share.
    """
    terms = []
    for position, row in enumerate(episode["observations"]):
        terms.append(
            Term(
                "local",
                "radio_dbm",
                row["radio_dbm"],
                position,
                # One phone signal measurement at each of three times: N = 3.
                config.local_weight / math.sqrt(3),
                factors_for("radio_dbm"),
                (row["source_id"],),
            )
        )
        if path:
            for field in CONTEXT_FIELDS:
                if field == "cell_load_pct":
                    ids = (row["source_id"], row["cell_state_source_id"])
                elif field.startswith("peer_"):
                    ids = (
                        row["source_id"],
                        row["edge_source_id"],
                        *row["peer_source_ids"],
                        *row.get("peer_edge_ids", []),
                    )
                else:
                    ids = (row["source_id"], row["edge_source_id"], row["link_state_source_id"])
                terms.append(
                    Term(
                        "context",
                        field,
                        row[field],
                        position,
                        # Five connected measurements at three times: N = 15.
                        # Default weight 2 gives 2/sqrt(15), about 0.516 per term,
                        # rather than twice the phone term's 1/sqrt(3), about 0.577.
                        config.context_weight / math.sqrt(15),
                        factors_for(field),
                        tuple(ids),
                    )
                )
        else:
            # Connectivity ablation: retain all same-time network measurements,
            # without selecting our phone's actual dependencies or peer averages.
            # It tests loss of graph selection as well as removal of path roles.
            pool = episode["network_pool"][position]
            count = sum(len(pool[field]) for field in CONTEXT_FIELDS)
            for field in CONTEXT_FIELDS:
                for value, identity in pool[field]:
                    terms.append(
                        Term(
                            "context",
                            field,
                            value,
                            position,
                            # The generator supplies count terms at each of three
                            # times, so the pool's total term count is 3*count.
                            config.context_weight / math.sqrt(3 * count),
                            ((f"attribute:{field}", 0),),
                            (identity,),
                        )
                    )
    if handset and config.handset_weight:
        # The phone model is constant across the episode: encode it once, with no
        # observation rotation or sqrt(3) divisor. It is a category, not a number.
        terms.append(
            Term(
                "handset",
                "handset",
                episode["handset"],
                0,
                config.handset_weight,
                (("attribute:handset", 0),),
                (episode["observations"][0]["phone_source_id"],),
            )
        )
    return terms


class Encoder:
    def __init__(self, config, seed):
        self.config, self.seed = config, seed
        self.dimension = config.dimension
        # One shared ordered codebook for every numeric field. Nearby levels are
        # more similar than distant ones; attribute binding distinguishes fields
        # that select the same level. No text embedding or fitted encoder is used.
        self.levels = torchhd.level(
            config.levels,
            config.dimension,
            vsa="MAP",
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )
        self.atom = lru_cache(maxsize=None)(self._atom)
        self.basis = lru_cache(maxsize=None)(self._basis)

    def _atom(self, name):
        # Hash the name together with the encoder seed to obtain a reproducible
        # generator seed. Built-in hash() can vary between processes. Truncation
        # and modulo fit the generator's seed range; they encode no domain order.
        seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}|{name}".encode()).digest()[:8], "little"
        ) % (2**63 - 1)
        return torchhd.random(
            1,
            self.dimension,
            vsa="MAP",
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
        )[0]

    def _basis(self, field, value_index, position, factors, component, binding, sequence):
        # Numeric facts select a correlated level; categories get independent
        # named atoms because "model_2" is not a numeric step above "model_1".
        value = (
            self.atom(f"value:{value_index}")
            if field in ("handset", "categorical")
            else self.levels[value_index]
        )
        if binding:
            # First permutation use: rotate each role by its structural position.
            # multibind multiplies these role arrays coordinate by coordinate;
            # bind then multiplies their product by the value hypervector.
            atoms = [torchhd.permute(self.atom(name), shifts=shift) for name, shift in factors]
            value = torchhd.bind(torchhd.multibind(torch.stack(atoms)), value)
        # A component marker separates phone signal, network context and handset
        # meanings in the same D coordinates; these are not concatenated sections.
        # The binding=False ablation omits factors above but keeps this marker.
        value = torchhd.bind(self.atom(f"channel:{component}"), value)
        # Second permutation use: rotate the whole bound fact by observation order.
        # Shifts 1/2/3 mean first/middle/last; +1 starts numbered events at shift 1.
        # This is the same rotation operator used above, not another permutation
        # family. It records order, not timestamps or the 20-second gaps. Handset
        # facts are timeless and receive no outer rotation.
        return (
            torchhd.permute(value, shifts=position + 1)
            if sequence and component != "handset"
            else value
        )

    def term_vector(self, term, binding=True, sequence=True):
        # Quantize the clipped fraction to the nearest of K indices, 0 through K-1.
        # K-1 is the number of intervals between endpoint levels, not K. Rounding
        # loses sub-level detail (Python round resolves exact ties to an even index).
        index = (
            term.value
            if term.field in ("handset", "categorical")
            else round(scaled(term.value, term.field) * (self.config.levels - 1))
        )
        return (
            self.basis(
                term.field,
                index,
                term.position,
                tuple(tuple(x) for x in term.factors),
                term.component,
                binding,
                sequence,
            )
            * term.weight
        )

    def encode_terms(self, terms, binding=True, sequence=True):
        if not terms:
            return torch.zeros(self.dimension, dtype=torch.float32)
        # TorchHD multiset is a coordinatewise sum for MAP. Preserve its float32
        # weighted accumulator: no majority-sign threshold and no normalization.
        # Retaining it allows subtraction of a component followed by normalization.
        return torchhd.multiset(
            torch.stack([self.term_vector(term, binding, sequence) for term in terms])
        ).as_subclass(torch.Tensor)

    def encode(self, episode, path=True, handset=True, binding=True, sequence=True):
        """Return the raw episode sum; unit() is applied later for comparison."""
        return self.encode_terms(terms_for(episode, self.config, path, handset), binding, sequence)

    def components(self, episode, **kwargs):
        """Return weighted raw component sums, using the existing internal keys."""
        terms = terms_for(
            episode, self.config, kwargs.get("path", True), kwargs.get("handset", True)
        )
        return {
            name: self.encode_terms(
                [t for t in terms if t.component == name],
                kwargs.get("binding", True),
                kwargs.get("sequence", True),
            )
            for name in ("local", "context", "handset")
        }

    def manifest(self, episode):
        return {
            "version": ENCODER_VERSION,
            "dimension": self.dimension,
            "seed": self.seed,
            "levels": self.config.levels,
            "ranges": RANGES,
            "config_hash": digest(self.config.to_dict()),
            "normalization": "L2 only after the float32 additive accumulator",
            "excluded_fields": EXCLUDED,
            "terms": [asdict(t) for t in terms_for(episode, self.config)],
        }

    def signature(self):
        return digest(
            {
                "version": ENCODER_VERSION,
                "seed": self.seed,
                "config": self.config.to_dict(),
                "levels_sha256": hashlib.sha256(self.levels.numpy().tobytes()).hexdigest(),
                "ranges": RANGES,
            }
        )


def model_input_features(episode, handset=True):
    """The 21 ordered, scaled inputs shared by LR and MLP."""
    # Three times * six numeric fields = 18 continuous inputs in fixed order.
    # The conventional classifiers use the same range scaling and joined facts,
    # but do not quantize values to the HDC codebook's 32 levels.
    values = [
        scaled(row[field], field)
        for row in episode["observations"]
        for field in (*LOCAL_FIELDS, *CONTEXT_FIELDS)
    ]
    # A continuous, ordered representation with precisely the same joined paths and numeric inputs.
    if handset:
        # Three one-hot category slots: the chosen model has value 0.25, others 0.
        # This matches the small handset coefficient, not HDC similarity geometry.
        values += [float(episode["handset"] == f"model_{i}") * 0.25 for i in range(3)]
    return torch.tensor(values, dtype=torch.float32)
