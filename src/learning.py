"""HDC memory, trained classifiers, and delayed review replay."""

import hashlib

import torch
import torch.nn.functional as F

from config import LABELS
from encoding import unit


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


class Prototype:
    def __init__(self, dimension):
        self.values = torch.zeros(len(LABELS), dimension, dtype=torch.float32)
        self.counts = torch.zeros(len(LABELS), dtype=torch.int64)
        self.history = []

    def scores(self, vector):
        return F.normalize(self.values, dim=1) @ unit(vector)

    def predict(self, vector):
        if not self.counts.sum():
            return None
        scores = self.scores(vector)
        scores[self.counts == 0] = -torch.inf
        return int(scores.argmax())

    def update(self, vector, label, record=None):
        previous = self.values[label].clone()
        before = tensor_hash(previous)
        self.values[label] += unit(vector)
        self.counts[label] += 1
        if record is not None:
            self.history.append(
                dict(
                    record,
                    label=LABELS[label],
                    previous_hash=before,
                    update_vector_hash=tensor_hash(unit(vector)),
                    updated_hash=tensor_hash(self.values[label]),
                )
            )
        # Exact reversal restores the previous float32 accumulator rather than subtracting rounded sums.
        return label, previous, record is not None

    def undo(self, undo):
        label, previous, recorded = undo
        self.values[label].copy_(previous)
        self.counts[label] -= 1
        if recorded:
            self.history.pop()


def classification_metrics(predictions, labels):
    matrix = torch.zeros(len(LABELS), len(LABELS), dtype=torch.int64)
    unavailable = 0
    for prediction, label in zip(predictions, labels):
        if prediction is None:
            unavailable += 1
        else:
            matrix[label, prediction] += 1
    per_class = {}
    for index, name in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = sum(int(label == index) for label in labels) - tp
        per_class[name] = {
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "recall": tp / max(1, tp + fn),
            "support": tp + fn,
        }
    return {
        "accuracy": float(matrix.diag().sum()) / max(1, len(labels)),
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(LABELS),
        "coverage": 1 - unavailable / max(1, len(labels)),
        "per_class": per_class,
        "confusion": matrix.tolist(),
        "insufficient_memory": unavailable,
    }


def budget_indices(indices, labels, budget, episodes):
    count = [0] * len(LABELS)
    selected = []
    for index in sorted(
        indices, key=lambda i: (episodes[i]["decision_s"], episodes[i]["episode_id"])
    ):
        label = int(labels[index])
        if count[label] < budget:
            selected.append(index)
            count[label] += 1
    if min(count) < budget:
        raise ValueError("Insufficient label coverage for a learning budget")
    return selected


def learning_curves(
    raws, features, labels, memory, testing, episodes, config, settings, seed=None, model_dir=None
):
    from pathlib import Path
    from time import perf_counter

    from classifiers import METHODS, fit_classifier, persist_classifier

    seed = config.encoder_seeds[0] if seed is None else seed
    if any(episodes[i]["split"] != "memory" for i in memory) or any(
        episodes[i]["split"] != "test" for i in testing
    ):
        raise ValueError(
            "Learning curves require memory-only reviews and independent final-test worlds"
        )
    output = []
    for budget in config.budgets:
        selected = budget_indices(memory, labels, budget, episodes)
        for method in ("hdc", *METHODS):
            if method == "hdc":
                model = Prototype(raws.shape[1])
                started = perf_counter()
                for index in selected:
                    model.update(raws[index], int(labels[index]))
                fit_info = {
                    "fit_ms": (perf_counter() - started) * 1000,
                    "parameters": "Fixed encoder and additive memory",
                    "convergence_warnings": [],
                }
                predictions = [model.predict(raws[index]) for index in testing]
            else:
                model, fit_info = fit_classifier(
                    method,
                    settings["parameters"][str(budget)][method],
                    seed,
                    features,
                    labels,
                    selected,
                )
                predictions = model.predict(features[testing].numpy()).tolist()
                if model_dir is not None:
                    fit_info["persistence"] = persist_classifier(
                        model,
                        features[testing],
                        Path(model_dir) / f"{method}-budget-{budget}.joblib",
                    )
            truth = labels[testing].tolist()
            by_world = {}
            for index, prediction, label in zip(testing, predictions, truth):
                by_world.setdefault(episodes[index]["world_id"], []).append((prediction, label))
            output.append(
                {
                    "budget_per_class": budget,
                    "method": method,
                    "metrics": classification_metrics(predictions, truth),
                    "fit": fit_info,
                    "per_world": {
                        key: classification_metrics([p for p, _ in pairs], [y for _, y in pairs])
                        for key, pairs in by_world.items()
                    },
                    "predictions": [
                        {
                            "episode_id": episodes[index]["episode_id"],
                            "truth": LABELS[int(labels[index])],
                            "prediction": LABELS[prediction] if prediction is not None else None,
                        }
                        for index, prediction in zip(testing, predictions)
                    ],
                    "review_episode_ids": [episodes[index]["episode_id"] for index in selected],
                }
            )
    return output


def delayed_feedback(raws, labels, memory, episodes, reviews, config, encoder_signature):
    models = {"hdc": Prototype(raws.shape[1])}
    order = sorted(memory, key=lambda i: (episodes[i]["decision_s"], episodes[i]["episode_id"]))
    pending, predictions, effects = [], {key: [] for key in models}, []
    sentinel = order[-min(20, len(order)) :]

    def apply(index, observed_at):
        model = models["hdc"]
        before = [model.predict(raws[j]) for j in sentinel]
        record = {
            "episode_id": episodes[index]["episode_id"],
            "review_source_id": reviews[episodes[index]["episode_id"]]["source_id"],
            "label_status": "simulated_evidence_rule",
            "review_ready_s": reviews[episodes[index]["episode_id"]]["ready_s"],
            "observed_at_s": observed_at,
            "encoder_hash": encoder_signature,
        }
        model.update(raws[index], int(labels[index]), record)
        after = [model.predict(raws[j]) for j in sentinel]
        changed = sum(a != b for a, b in zip(before, after))
        corrected = sum(
            a != int(labels[j]) and b == int(labels[j]) for a, b, j in zip(before, after, sentinel)
        )
        regressed = sum(
            a == int(labels[j]) and b != int(labels[j]) for a, b, j in zip(before, after, sentinel)
        )
        effects.append(
            dict(
                record,
                changed=changed,
                corrected=corrected,
                regressed=regressed,
                net_correct=corrected - regressed,
                probe_scope="Fixed diagnostic memory-partition episodes; probes never select updates",
            )
        )

    for index in order:
        now = episodes[index]["decision_s"]
        ready = sorted(
            [j for j in pending if reviews[episodes[j]["episode_id"]]["ready_s"] <= now],
            key=lambda j: (
                reviews[episodes[j]["episode_id"]]["ready_s"],
                episodes[j]["episode_id"],
            ),
        )
        for j in ready:
            apply(j, now)
            pending.remove(j)
        for method, model in models.items():
            prediction = model.predict(raws[index])
            predictions[method].append(
                {
                    "episode_id": episodes[index]["episode_id"],
                    "decision_s": now,
                    "truth": LABELS[int(labels[index])],
                    "prediction": LABELS[prediction] if prediction is not None else None,
                    "reviews_available": int(models["hdc"].counts.sum()),
                }
            )
        pending.append(index)
    for index in sorted(
        pending,
        key=lambda j: (reviews[episodes[j]["episode_id"]]["ready_s"], episodes[j]["episode_id"]),
    ):
        apply(index, reviews[episodes[index]["episode_id"]]["ready_s"])
    summary = {}
    for method, rows in predictions.items():
        p = [
            LABELS.index(row["prediction"]) if row["prediction"] is not None else None
            for row in rows
        ]
        truth = [LABELS.index(row["truth"]) for row in rows]
        summary[method] = classification_metrics(p, truth)
    # Demonstrate exact reversibility without changing the returned final memory.
    previous = models["hdc"].values.clone()
    undo = models["hdc"].update(raws[order[0]], int(labels[order[0]]))
    models["hdc"].undo(undo)
    if not torch.equal(previous, models["hdc"].values):
        raise AssertionError("Prototype reversal was not exact")
    return {
        "metrics": summary,
        "predictions": predictions,
        "updates": models["hdc"].history,
        "effects": effects,
        "encoder_hash": encoder_signature,
        "reversal_exact": True,
        "review_delay_s": config.review_delay_s,
        "scope": "Predict-then-review replay within the memory partition; validation and test labels never update memory",
    }


def replay_updates(raw_by_id, updates, dimension):
    """Recover persisted memory or its pre-update state by replaying a log prefix."""
    model = Prototype(dimension)
    for record in updates:
        if record["review_ready_s"] > record["observed_at_s"]:
            raise ValueError("Review applied before availability")
        label = LABELS.index(record["label"])
        vector = raw_by_id[record["episode_id"]]
        if tensor_hash(model.values[label]) != record["previous_hash"]:
            raise ValueError("Audit replay predecessor mismatch")
        if tensor_hash(unit(vector)) != record["update_vector_hash"]:
            raise ValueError("Audit replay vector mismatch")
        model.update(vector, label)
        if tensor_hash(model.values[label]) != record["updated_hash"]:
            raise ValueError("Audit replay update mismatch")
    return model
