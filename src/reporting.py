"""Readable evidence, exportable figures, and measured claim boundaries."""

import collections
import json
import math
import os
import re
import statistics
from pathlib import Path

from config import LABELS, ROOT, Config, file_hash, save_json

COLOURS = {
    "hdc": "#287b78",
    "logistic_regression": "#a06935",
    "mlp": "#7461a2",
    "hdc_without_paths": "#929eaa",
    "hdc_without_order": "#7461a2",
}
NAMES = {
    "hdc": "HDC",
    "hdc_without_paths": "HDC: connectivity omitted",
    "hdc_without_order": "HDC: order omitted",
    "hdc_lance_float16": "HDC in Lance (float16)",
    "logistic_regression": "Trained logistic regression",
    "mlp": "Small trained MLP",
}
PATTERNS = {
    "radio_deteriorating": "deteriorating radio",
    "transient_recovery": "transient disruption and recovery",
    "shared_transport": "shared transport impairment",
    "normal": "normal service",
}


UPDATE_METHODS = ("hdc", "logistic_regression", "mlp")
UPDATE_NAMES = {
    "hdc": "HDC addition",
    "logistic_regression": "LR retraining",
    "mlp": "MLP retraining",
}

REPORT_TEMPLATES = ROOT / "docs" / "report-templates"
PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")


def render_template(name, values=None):
    """Fill named Markdown placeholders once, preserving LaTeX and inserted content."""
    template = (REPORT_TEMPLATES / name).read_text()
    values = {} if values is None else values
    missing = set(PLACEHOLDER.findall(template)) - values.keys()
    if missing:
        raise ValueError(f"Missing report values in {name}: {', '.join(sorted(missing))}")
    return PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)


def read(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return f"{value * 100:.1f}%"


def interval_text(value):
    return f"{percent(value['mean'])} [{percent(value['lower'])}, {percent(value['upper'])}]"


def timing_median(results, key, metric="median_ms"):
    """Median across seed pairs of one per-call benchmark."""
    return statistics.median(result["resources"]["timings"][key][metric] for result in results)


def update_rows(results, method, batch):
    return [
        row
        for result in results
        for row in result["resources"]["learning_updates"]["rows"]
        if row["method"] == method and row["batch_size"] == batch
    ]


def update_median(results, method, batch, stage="complete_wall_ms", metric="median_ms"):
    """Median across seed pairs of the raw (not per-review) batch timing."""
    return statistics.median(
        row["timings"][stage][metric] for row in update_rows(results, method, batch)
    )


def update_batches(results):
    return sorted(
        {
            row["batch_size"]
            for result in results
            for row in result["resources"]["learning_updates"]["rows"]
        }
    )


def ms(value):
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def figure_style():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/telus-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/telus-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelcolor": "#344454",
            "text.color": "#263647",
            "axes.edgecolor": "#c3cbd2",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#f4f6f8",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.2,
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(figure, root, name, plt):
    figure.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    # The report links PNG charts; remove legacy duplicate exports on reruns.
    (root / f"{name}.svg").unlink(missing_ok=True)
    plt.close(figure)


def make_schema_figure(root, plt):
    """A conceptual projection of the joined records, with illustrative values."""
    import io

    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    background, ink, muted, line, orange = "#FAF9F6", "#15141A", "#5A5864", "#E2DFD8", "#CD6F3E"
    figure, axis = plt.subplots(figsize=(16, 10.8))
    figure.patch.set_facecolor(background)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.set(xlim=(0, 1600), ylim=(1080, 0))
    axis.set_axis_off()
    axis.text(
        55,
        55,
        "From one phone to a shared network dependency",
        fontsize=23,
        weight="bold",
        color=ink,
    )
    axis.text(
        55,
        92,
        "Conceptual graph schema · example values at observation time t",
        fontsize=14,
        color=muted,
    )

    def properties(x, y, entries, width=224):
        height = 18 + 25 * len(entries)
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=6",
                facecolor="white",
                edgecolor=line,
                linewidth=1.2,
                zorder=4,
            )
        )
        axis.text(
            x + 12,
            y + 10,
            "\n".join(entries),
            va="top",
            fontsize=11.5,
            fontfamily="DejaVu Sans Mono",
            linespacing=1.45,
            color=muted,
            zorder=5,
        )

    def node(x, y, name, entries, active=False):
        axis.add_patch(
            Circle(
                (x, y),
                77,
                facecolor=background,
                edgecolor=orange if active else ink,
                linewidth=2,
                zorder=3,
            )
        )
        axis.text(
            x, y, name, ha="center", va="center", fontsize=14, weight="bold", color=ink, zorder=4
        )
        # A short connector ties each property box to the node's bottom-right corner.
        axis.plot([x + 50, x + 50], [y + 59, y + 95], color=line, linewidth=1.2, zorder=2)
        properties(x + 18, y + 95, entries)

    def edge(start, end, label, label_x, label_y, entries, active=False):
        colour = orange if active else muted
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=2.2 if active else 1.7,
                color=colour,
                shrinkA=0,
                shrinkB=2,
                zorder=1,
            )
        )
        axis.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=12.5,
            weight="bold",
            color=colour,
            zorder=5,
        )
        properties(label_x - 98, label_y + 10, entries, width=196)

    # Orange identifies the selected commuter path; arrows denote relationships, not causes.
    edge((187, 260), (393, 260), "HAS_PHONE", 290, 175, ["subscriber_id: S7"])
    edge((547, 260), (773, 260), "SERVED_BY", 660, 175, ["timestamp_s: t"], active=True)
    edge(
        (927, 260),
        (1173, 260),
        "USES",
        1050,
        150,
        ["valid_from_s: t0", "valid_to_s: t1"],
        active=True,
    )
    edge((187, 780), (393, 780), "HAS_OBSERVATION", 290, 695, ["ordinal: 1"])
    edge((470, 703), (470, 337), "OF_PHONE", 340, 510, ["phone_id: P7"])
    edge((927, 780), (1173, 780), "AT_CELL", 1050, 695, ["cell_id: C5"])
    edge((1250, 703), (1250, 337), "USES", 1370, 510, ["valid_from_s: t0", "valid_to_s: t1"])

    node(110, 260, "Subscriber", ["subscriber_id: S7", "pseudonym: Traveller7"])
    node(470, 260, "Phone", ["phone_id: P7", "handset: model_1"], active=True)
    node(850, 260, "Serving\ncell", ["cell_id: C2", "load_pct(t): 62"], active=True)
    node(
        1250,
        260,
        "Backhaul\nlink",
        ["link_id: L0", "loss_pct(t): 4.8", "latency_ms(t): 60"],
        active=True,
    )
    node(110, 780, "Episode", ["episode_id: E7", "observations: 3"])
    node(
        470,
        780,
        "Observation",
        ["observation_id: O1", "timestamp_s: t", "radio_dbm: -104", "geometry: Point(x,y)"],
    )
    node(
        850,
        780,
        "Peer\nobservation",
        ["peer_obs_id: PO1", "timestamp_s: t", "peer_radio_dbm: -82", "peer_loss_pct: 4.7"],
    )
    node(1250, 780, "Peer's\ncell", ["cell_id: C5"])

    axis.text(
        60, 1020, "Orange: commuter dependency path", fontsize=13, color=orange, weight="bold"
    )
    axis.text(
        510,
        1020,
        "Both cells use L0 at t: peers can share an upstream problem.",
        fontsize=13,
        color=ink,
    )
    axis.text(
        60,
        1053,
        "Circles are nodes; arrows are edges. Property boxes contain illustrative key-value pairs.",
        fontsize=12,
        color=muted,
    )
    # Keep HTML as the editable display source; SVG and PNG are matching figure exports.
    with plt.rc_context({"svg.fonttype": "none"}):
        output = io.StringIO()
        figure.savefig(output, format="svg", facecolor=background)
    svg = output.getvalue()
    svg = svg.replace(
        "font-family: 'DejaVu Sans';", "font-family: 'DejaVu Sans', Arial, sans-serif;"
    )
    svg = svg.replace(
        "font-family: 'DejaVu Sans Mono';",
        "font-family: 'DejaVu Sans Mono', 'Courier New', monospace;",
    )
    svg = svg.replace(
        "<svg ", '<svg role="img" aria-labelledby="telecom-schema-title telecom-schema-desc" ', 1
    )
    start = svg.index(">", svg.index("<svg ")) + 1
    svg = (
        svg[:start]
        + """
 <title id="telecom-schema-title">Conceptual telecom incident graph schema</title>
 <desc id="telecom-schema-desc">A subscriber has a phone. An episode contains ordered observations of that phone.
 At time t the phone is served by cell C2, which uses backhaul link L0. A peer observation at cell C5
 shares L0 through another time-valid edge. Node property boxes sit at the bottom right; edge property
 boxes sit below the edge labels. Cell and link measurements are joined telemetry at t.</desc>"""
        + svg[start:]
    )
    html = (
        '<!doctype html><html lang="en-CA"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Telecom incident graph schema</title><style>body{margin:0;background:#FAF9F6}.figure{overflow:auto}svg{display:block;width:100%;min-width:1100px;height:auto}</style><div class="figure">'
        + svg[svg.index("<svg ") :]
        + "</div></html>"
    )
    (root / "schema.html").write_text(html)
    (root / "schema.svg").write_text(svg)
    figure.savefig(root / "schema.png", dpi=180, facecolor=background)
    plt.close(figure)


def make_retrieval_figure(root, summary, plt):
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    methods = ["hdc", "hdc_without_paths", "hdc_without_order"]
    points = [summary["retrieval"][method]["precision_at_5"] for method in methods]
    means = [point["mean"] for point in points]
    axis.barh(
        range(len(methods)), means, color=[COLOURS[method] for method in methods], height=0.62
    )
    axis.errorbar(
        means,
        range(len(methods)),
        xerr=[
            [point["mean"] - point["lower"] for point in points],
            [point["upper"] - point["mean"] for point in points],
        ],
        fmt="none",
        ecolor="#243342",
        capsize=4,
    )
    axis.set_yticks(range(len(methods)), [NAMES[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.06)
    axis.set_xlabel("Share of the first five retrieved incidents with the same triage pattern")
    axis.set_title("Order and connected context improve incident retrieval", loc="left")
    for index, mean in enumerate(means):
        axis.text(min(mean + 0.018, 1.01), index, percent(mean), va="center", fontsize=10)
    axis.axvline(
        summary["random_precision_at_5"],
        color="#637181",
        linestyle="--",
        label="Random ranking reference",
    )
    axis.legend(loc="lower right", frameon=False)
    save_figure(figure, root, "retrieval", plt)


def make_figures(root, summary, results, case, config, geography_root):
    plt = figure_style()
    root.mkdir(parents=True, exist_ok=True)
    make_schema_figure(root, plt)
    make_retrieval_figure(root, summary, plt)

    figure, axis = plt.subplots(figsize=(8.6, 5), layout="constrained")
    for method in ("hdc", "logistic_regression", "mlp"):
        rows = [summary["learning"][str(budget)][method] for budget in config.budgets]
        means = [row["mean"] for row in rows]
        axis.plot(
            config.budgets,
            means,
            marker="o",
            linewidth=2.4,
            color=COLOURS[method],
            label=NAMES[method],
        )
        axis.fill_between(
            config.budgets,
            [row["lower"] for row in rows],
            [row["upper"] for row in rows],
            color=COLOURS[method],
            alpha=0.12,
        )
    axis.set_xscale("log")
    axis.set_xticks(config.budgets, [str(budget) for budget in config.budgets])
    from matplotlib.ticker import MaxNLocator, NullFormatter

    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Reviewed examples per class (four classes)")
    axis.set_ylabel("Macro F1 on independent final-test worlds")
    axis.set_title("Memory starts empty and grows through reviews", loc="left")
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure, root, "learning", plt)

    # Raw batch totals only; dividing by batch size would mix throughput with latency.
    batches = update_batches(results)
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for method in UPDATE_METHODS:
        totals = [update_median(results, method, batch) for batch in batches]
        axis.plot(
            batches,
            totals,
            "-o",
            linewidth=2.4,
            color=COLOURS[method],
            label=UPDATE_NAMES[method],
        )
        for batch, total in zip(batches, totals):
            axis.annotate(
                f"{total:.2f} ms" if total < 10 else f"{total:.1f} ms",
                (batch, total),
                textcoords="offset points",
                # HDC rises steeply, so label below its points to keep them off the line,
                # except where it ends above LR retraining.
                xytext=(8, -14) if method == "hdc" and batch != max(batches) else (8, 4),
                fontsize=9,
                color=COLOURS[method],
            )
    axis.set_xticks(batches)
    axis.set_xlim(min(batches) - 3, max(batches) + 8)
    axis.set_yscale("log")
    axis.set_ylim(bottom=min(update_median(results, "hdc", batch) for batch in batches) / 1.6)
    axis.set_xlabel("Newly reviewed incidents arriving together")
    axis.set_ylabel("Total time to encode, learn and predict, ms (log scale)")
    initial = results[0]["resources"]["learning_updates"]["rows"][0]["initial_training_examples"]
    axis.set_title(
        f"Learning from new reviews, starting from the same {initial} reviewed incidents",
        loc="left",
    )
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure, root, "learning-updates", plt)

    from geography import load_geography

    lines, _ = load_geography(geography_root)
    query = case["query"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    axis = axes[0]
    for line in lines:
        axis.plot([p[0] for p in line], [p[1] for p in line], color="#b9c1c8", linewidth=0.85)
    observations = query["observations"]
    xs, ys = [row["longitude"] for row in observations], [row["latitude"] for row in observations]
    axis.plot(xs, ys, color="#287b78", marker="o", markersize=6, linewidth=2)
    axis.set_xlim(min(xs) - 0.007, max(xs) + 0.007)
    axis.set_ylim(min(ys) - 0.003, max(ys) + 0.003)
    axis.set_title("1. A simulated commute", loc="left")
    axis.set_xlabel("Longitude (public road geometry)")
    axis.set_ylabel("Latitude")
    axis.ticklabel_format(useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis = axes[1]
    axis.plot(
        [0, 20, 40],
        [row["radio_dbm"] for row in observations],
        "-o",
        color="#287b78",
        label="Received signal strength",
    )
    axis.set_ylim(-125, -65)
    axis.set_xticks([0, 20, 40], ["Before", "During", "After"])
    axis.set_ylabel("Received signal strength (dBm)")
    axis.set_title("2. Three observations", loc="left")
    axis = axes[2]
    mid = observations[1]
    axis.grid(False)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal")
    from matplotlib.patches import Circle

    nodes = [
        (0.17, 0.8, "Phone"),
        (0.73, 0.8, "Serving\ncell"),
        (0.73, 0.36, "Backhaul\nlink"),
        (0.17, 0.36, "Peer\nphones"),
    ]
    for x, y, text in nodes:
        axis.add_patch(
            Circle((x, y), 0.135, facecolor="#e3efed", edgecolor="#639b97", linewidth=1.5)
        )
        axis.text(x, y, text, ha="center", va="center", fontsize=9)
    for a, b in [
        ((0.305, 0.8), (0.595, 0.8)),
        ((0.73, 0.665), (0.73, 0.495)),
        ((0.305, 0.36), (0.595, 0.36)),
    ]:
        axis.annotate(
            "", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#344454", "lw": 1.6}
        )
    axis.text(0.97, 0.56, "time-valid\nedge", ha="right", fontsize=9)
    axis.text(
        0.5,
        0.08,
        f"Link loss: {mid['link_loss_pct']:.1f}%\nPeer loss: {mid['peer_loss_pct']:.1f}%",
        ha="center",
        fontsize=10,
    )
    axis.set_title("3. Inspect connected facts", loc="left")
    figure.suptitle(
        "Public streets; every phone, network edge and measurement is simulated", fontsize=13
    )
    save_figure(figure, root, "incident", plt)


def learning_update_explanation(results, config):
    batches = update_batches(results)
    totals = {
        method: {batch: update_median(results, method, batch) for batch in batches}
        for method in UPDATE_METHODS
    }
    rows = "\n".join(
        f"| {batch} | " + " | ".join(ms(totals[method][batch]) for method in UPDATE_METHODS) + " |"
        for batch in batches
    )
    single, largest = min(batches), max(batches)
    stages = {
        method: {
            key: update_median(results, method, single, key)
            for key in ("encoding_ms", "learning_ms", "prediction_ms")
        }
        for method in UPDATE_METHODS
    }
    stage_rows = "\n".join(
        f"| {UPDATE_NAMES[method]} | "
        + " | ".join(
            ms(stages[method][key]) for key in ("encoding_ms", "learning_ms", "prediction_ms")
        )
        + f" | {ms(totals[method][single])} |"
        for method in UPDATE_METHODS
    )

    def share(method, key):
        # Floor so that, for example, 99.6% is not reported as the whole time.
        return math.floor(100 * stages[method][key] / sum(stages[method].values()))

    hdc, lr, mlp = (totals[method] for method in UPDATE_METHODS)
    review = "review" if single == 1 else "reviews"
    finding = (
        f"**Result.** For {single} new {review}, HDC finishes in {ms(hdc[single])} ms: "
        f"about {lr[single] / hdc[single]:.0f}× faster than retraining LR ({ms(lr[single])} ms) and "
        f"{mlp[single] / hdc[single]:.0f}× faster than retraining the MLP ({ms(mlp[single])} ms). "
        f"Encoding is {share('hdc', 'encoding_ms'):}% of HDC's time, while retraining is "
        f"{share('logistic_regression', 'learning_ms'):}% of LR's and {share('mlp', 'learning_ms'):}% of the MLP's."
    )
    if largest > single:
        slope = (hdc[largest] - hdc[single]) / (largest - single)
        faster = [
            UPDATE_NAMES[method].split()[0]
            for method in ("logistic_regression", "mlp")
            if totals[method][largest] < hdc[largest]
        ]
        crossover = (
            f"At {largest} reviews, retraining {' and '.join(faster)} once is faster than {largest} HDC additions "
            f"({ms(hdc[largest])} ms)."
            if faster
            else f"At {largest} reviews, HDC ({ms(hdc[largest])} ms) is still faster than retraining either model."
        )
        finding += (
            f"\n\nBatch size changes the comparison. HDC's total grows with every review it adds, "
            f"by about {slope:.2f} ms each. Retraining processes all retained reviews whatever the batch size, "
            f"so its time depends little on how many reviews arrived: LR takes {ms(lr[single])} ms for {single} and "
            f"{ms(lr[largest])} ms for {largest}; the MLP takes {ms(mlp[single])} and {ms(mlp[largest])} ms. {crossover}"
        )
    warning_count = sum(
        bool(info["convergence_warnings"])
        for result in results
        for row in result["resources"]["learning_updates"]["rows"]
        for info in row["fit_diagnostics"]
    )
    initial = results[0]["resources"]["learning_updates"]["rows"][0]["initial_training_examples"]
    skipped = results[0]["resources"]["learning_updates"]["skipped_batches"]
    skipped_text = (
        " The small run omits batches without enough later memory reviews: "
        + ", ".join(str(row["batch_size"]) for row in skipped)
        + "."
        if skipped
        else ""
    )
    sizes = [str(batch) for batch in batches]
    return render_template(
        "learning-updates.md",
        {
            "initial_reviews": initial,
            "max_reviews_per_class": max(config.budgets),
            "batch_sizes": ", ".join(sizes[:-1]) + " or " + sizes[-1]
            if len(sizes) > 1
            else sizes[0],
            "refit_sizes": ", ".join(str(initial + b) for b in batches[:-1])
            + (" or " if len(batches) > 1 else "")
            + str(initial + batches[-1]),
            "refit_warmup": config.refit_warmup,
            "refit_repeats": config.refit_repeats,
            "batch_coverage": skipped_text,
            "learning_update_rows": rows,
            "learning_update_stage_rows": stage_rows,
            "learning_update_finding": finding,
            "refit_warning_count": warning_count,
        },
    )


def encoder_explanation(config):
    return render_template(
        "encoder.md",
        {
            "dimension": f"{config.dimension:,}",
            "levels": config.levels,
            "phone_signal_weight": config.local_weight,
            "context_weight": config.context_weight,
            "handset_weight": config.handset_weight,
        },
    )


def geographic_enhancement_explanation():
    return render_template("geographic-enhancements.md")


# Short everyday names for the patterns, used in the future-work diagnosis.
PLAIN_NAMES = {
    "radio_deteriorating": "signal getting weaker",
    "transient_recovery": "signal recovering",
    "shared_transport": "shared-equipment fault",
    "normal": "normal service",
}
PLAIN_INTROS = {
    "radio_deteriorating": "a phone's signal getting steadily weaker",
    "transient_recovery": "a phone's signal dropping and then recovering",
    "shared_transport": "faults in network equipment that many phones share (*shared transport impairment* in Experiment 3)",
    "normal": "normal service",
}


def future_work_explanation(run_dir, config, pair_dir, data_seed, encoder_seed, budget, hardest):
    """Rebuild one pair's class memory to show where and why its mistakes happen."""
    import joblib
    import torch
    import torch.nn.functional as F

    from data import build_episodes, load_tables
    from encoding import Encoder, model_input_features, scaled
    from learning import Prototype, budget_indices

    tables = load_tables(run_dir / "data" / str(data_seed))
    episodes = build_episodes(tables)
    reviews = {row["episode_id"]: row for row in tables["reviews"].to_pylist()}
    labels = torch.tensor([LABELS.index(reviews[e["episode_id"]]["label"]) for e in episodes])
    encoder = Encoder(config, encoder_seed)
    raws = torch.stack([encoder.encode(episode) for episode in episodes])
    memory = [i for i, e in enumerate(episodes) if e["split"] == "memory"]
    testing = [i for i, e in enumerate(episodes) if e["split"] == "test"]
    hdc = Prototype(raws.shape[1])
    for index in budget_indices(memory, labels, budget, episodes):
        hdc.update(raws[index], int(labels[index]))
    features = torch.stack([model_input_features(episode) for episode in episodes])
    lr = joblib.load(pair_dir / "models" / f"logistic_regression-budget-{budget}.joblib")
    lr_predictions = dict(zip(testing, lr.predict(features[testing].numpy()).tolist()))

    # The review rule's 15 dB change separates weakening, recovering and steady signal.
    threshold = 15

    def signal(episode):
        change = episode["observations"][-1]["radio_dbm"] - episode["observations"][0]["radio_dbm"]
        return (
            "weakens"
            if change <= -threshold
            else "recovers"
            if change >= threshold
            else "stays steady"
        )

    target = LABELS.index(hardest)
    hard = [i for i in testing if labels[i] == target]
    wrong = collections.Counter(
        hdc.predict(raws[i]) for i in hard if hdc.predict(raws[i]) != target
    )
    confused = wrong.most_common(1)[0][0]
    groups = {}
    for i in hard:
        row = groups.setdefault(signal(episodes[i]), [0, 0, 0])
        row[0] += 1
        row[1] += hdc.predict(raws[i]) != target
        row[2] += lr_predictions[i] != target
    order = ("weakens", "stays steady", "recovers")
    group_rows = "\n".join(
        f"| {name.capitalize()} | {n} | {percent(h / n)} | {percent(l / n)} |"
        for name in order
        if name in groups
        for n, h, l in [groups[name]]
    )
    worst = max(groups, key=lambda name: groups[name][1] / groups[name][0])
    steadiest = min(groups, key=lambda name: groups[name][1] / groups[name][0])

    # Split each score exactly into channels: unit(z) . memory = sum of channel . memory / |z|.
    memories = F.normalize(hdc.values, dim=1)
    names = {
        "local": "Phone's own signal",
        "context": "Network facts (links, peers, cell load)",
        "handset": "Handset model",
    }
    split = {name: [0.0, 0.0] for name in names}
    members = [i for i in hard if signal(episodes[i]) == worst]
    for i in members:
        norm = raws[i].norm()
        for name, vector in encoder.components(episodes[i]).items():
            split[name][0] += float(vector @ memories[target] / norm) / len(members)
            split[name][1] += float(vector @ memories[confused] / norm) / len(members)
    split_rows = "\n".join(
        f"| {names[name]} | {a:.3f} | {b:.3f} | {a - b:+.3f} |" for name, (a, b) in split.items()
    )
    margin = sum(a - b for a, b in split.values())

    level_finding = ""
    if hardest == "shared_transport":
        # Compare the mildest faulty link with the worst healthy one, as encoded levels.
        faulty = [
            episodes[i]["observations"][1]["link_loss_pct"]
            for i in range(len(episodes))
            if labels[i] == target
        ]
        healthy = [
            observation["link_loss_pct"]
            for i, episode in enumerate(episodes)
            for position, observation in enumerate(episode["observations"])
            if labels[i] != target or position != 1
        ]
        low, high = max(healthy), min(faulty)

        def level(value):
            return round((config.levels - 1) * scaled(value, "link_loss_pct"))

        cosine = float(
            F.cosine_similarity(encoder.levels[level(low)], encoder.levels[level(high)], dim=0)
        )
        level_finding = (
            f"**Why the network evidence is weak.** The deciding facts are a small share of the encoding, and a faulty "
            f"link does not look very different from a healthy one. The mildest fault in the data ({high:.1f}% packet loss) "
            f"and the worst healthy link ({low:.1f}%) fall on numeric levels {level(high)} and {level(low)}, whose encodings "
            f"have cosine similarity {cosine:.2f}: nearby numbers are deliberately encoded alike."
        )
    worst_rate, steady_rate = (groups[name][1] / groups[name][0] for name in (worst, steadiest))
    finding = (
        f"when the phone's own signal {worst}, HDC mislabels {percent(worst_rate)} of these incidents, against "
        f'{percent(steady_rate)} when it {steadiest}. Its most common wrong answer is "{PLAIN_NAMES[LABELS[confused]]}".'
    )
    return render_template(
        "future-work.md",
        {
            "hard_pattern_intro": PLAIN_INTROS[hardest],
            "hard_pattern": PLAIN_NAMES[hardest],
            "hard_pattern_finding": finding,
            "data_seed": data_seed,
            "encoder_seed": encoder_seed,
            "budget": budget,
            "radio_threshold": threshold,
            "group_rows": group_rows,
            "split_count": len(members),
            "split_group": worst,
            "confused_pattern": PLAIN_NAMES[LABELS[confused]],
            "split_rows": split_rows,
            "net_margin": f"{margin:.3f}",
            "level_finding": level_finding,
        },
    )


def learning_comparison(summary, config, class_errors, error_budget):
    """Lead Experiment 3 with what the paired differences support at each budget."""
    names = {"logistic_regression": "LR", "mlp": "the MLP"}
    budgets = sorted(config.budgets)

    def difference(budget, method):
        return summary["learning_differences"][str(budget)]["hdc_minus_" + method]

    def clear(value):
        # Decide on the displayed one-decimal bounds, so a bold "[0.0, ...]" never appears.
        return round(value["lower"] * 100, 1) > 0 or round(value["upper"] * 100, 1) < 0

    def cell(value):
        text = (
            f"{value['mean'] * 100:+.1f} [{value['lower'] * 100:.1f}, {value['upper'] * 100:.1f}]"
        )
        return f"**{text}**" if clear(value) else text

    rows = "\n".join(
        f"| {budget} | "
        + " | ".join(
            percent(summary["learning"][str(budget)][method]["mean"])
            for method in ("hdc", "logistic_regression", "mlp")
        )
        + " | "
        + " | ".join(cell(difference(budget, method)) for method in names)
        + " |"
        for budget in budgets
    )

    first = budgets[0]
    first_diffs = {method: difference(first, method) for method in names}
    reviews = "review" if first == 1 else "reviews"
    hdc_first = percent(summary["learning"][str(first)]["hdc"]["mean"])
    ahead = all(clear(value) and value["mean"] > 0 for value in first_diffs.values())
    gaps = [first_diffs[method]["mean"] * 100 for method in names]
    if ahead:
        opening = (
            f"**Result.** With {first} {reviews} per class ({first * 4} in total), HDC reaches {hdc_first} macro F1: "
            f"{gaps[0]:.1f} points above LR and {gaps[1]:.1f} above the MLP, and both intervals exclude zero."
        )
    else:
        opening = (
            f"**Result.** With {first} {reviews} per class ({first * 4} in total), HDC reaches {hdc_first} macro F1, against "
            + " and ".join(
                f"{percent(summary['learning'][str(first)][method]['mean'])} for {name}"
                for method, name in names.items()
            )
            + "."
        )

    # On par: from this budget onward, the three mean scores stay within 2.5 points.
    def spread(budget):
        scores = [
            summary["learning"][str(budget)][method]["mean"] * 100
            for method in ("hdc", "logistic_regression", "mlp")
        ]
        return max(scores) - min(scores)

    on_par = next(
        (
            budget
            for budget in budgets
            if all(spread(later) <= 2.5 for later in budgets if later >= budget)
        ),
        None,
    )
    sentences = [opening]
    if on_par is not None and on_par > first:
        sentences.append(
            f"From {on_par} reviews per class onward, all three are within "
            f"{max(spread(budget) for budget in budgets if budget >= on_par):.1f} points of one another."
        )
        small = []
        for budget in budgets:
            if budget < on_par:
                continue
            for method, name in names.items():
                value = difference(budget, method)
                if clear(value):
                    leader, other = ("HDC", name) if value["mean"] > 0 else (name, "HDC")
                    small.append(
                        f"{leader} ahead of {other} by {abs(value['mean']) * 100:.1f} points at {budget}"
                    )
        if small:
            sentences.append(
                "Within that range, a few small differences are distinguishable from zero: "
                + "; ".join(small)
                + " reviews per class."
            )
    headline = " ".join(sentences)

    if ahead and on_par is not None and on_par > first:
        takeaway = (
            f"HDC learns most efficiently when labels are scarce: with {first} {reviews} per class, it scores "
            f"{min(gaps):.1f}–{max(gaps):.1f} points higher macro F1 than LR and the MLP. From {on_par} reviews per class, "
            "the three are on par."
        )
    else:
        takeaway = (
            "Experiment 3 compares HDC's macro F1 with LR and the MLP at every review budget."
        )

    hardest = max(LABELS, key=lambda label: class_errors["hdc"][label])
    others = [label for label in LABELS if label != hardest]
    pattern = (
        f"Averages can hide a harder pattern. At {error_budget} reviews per class, HDC assigns "
        f"{percent(class_errors['hdc'][hardest])} of {PATTERNS[hardest]} incidents to the wrong pattern, against "
        f"{percent(class_errors['logistic_regression'][hardest])} for LR and {percent(class_errors['mlp'][hardest])} for the MLP; "
        f"its error rate on every other pattern is at most {percent(max(class_errors['hdc'][label] for label in others))}. "
        "The per-pattern table is in the details below."
    )
    facts = {
        "first": first,
        "ahead": ahead,
        "gaps": gaps,
        "on_par": on_par,
        "spread": max(spread(b) for b in budgets if b >= on_par) if on_par else None,
        "hardest": hardest,
    }
    return rows, headline, takeaway, pattern, facts


def create_report(run_dir):
    from study import code_hash, verify_prepared

    run_dir = Path(run_dir)
    config = Config.read(run_dir / "config.json")
    manifest = read(run_dir / "manifest.json")
    verify_prepared(run_dir, config)
    if manifest["stage"] != "complete" or len(manifest["completed_pairs"]) != len(
        config.data_seeds
    ) * len(config.encoder_seeds):
        raise ValueError("A complete experiment grid is required before reporting")
    if manifest["run_code_hash"] != code_hash() or manifest["summary_hash"] != file_hash(
        run_dir / "summary.json"
    ):
        raise ValueError("Code or summary changed after the recorded experiment")
    for name, checksum in manifest["artifact_hashes"].items():
        if file_hash(run_dir / name) != checksum:
            raise ValueError(f"Report artifact changed: {name}")
    paths = [
        run_dir / "pairs" / f"data-{data}_encoder-{encoder}"
        for data in config.data_seeds
        for encoder in config.encoder_seeds
    ]
    for path, (data, encoder) in zip(
        paths, [(d, e) for d in config.data_seeds for e in config.encoder_seeds]
    ):
        if file_hash(path / "results.json") != manifest["result_hashes"][f"{data}/{encoder}"]:
            raise ValueError("Pair result checksum mismatch")
    results = [read(path / "results.json") for path in paths]
    summary = read(run_dir / "summary.json")
    case = read(paths[0] / "case.json")
    online = read(paths[0] / "online.json")
    geography = read(run_dir / "data" / str(config.data_seeds[0]) / "manifest.json")["geography"]
    root = run_dir / "reports"
    source_link = os.path.relpath((ROOT / "src").resolve(), root.resolve())
    make_figures(root / "figures", summary, results, case, config, run_dir / "geography")
    retrieval_rows = "\n".join(
        f"| {NAMES[method]} | {interval_text(summary['retrieval'][method]['precision_at_5'])} | {percent(summary['retrieval'][method]['top1']['mean'])} | {summary['retrieval'][method]['reciprocal_rank_at_10']['mean']:.3f} |"
        for method in (
            "hdc",
            "hdc_without_paths",
            "hdc_without_order",
            "hdc_lance_float16",
        )
    )
    error_budget = 5 if 5 in config.budgets else min(config.budgets)
    class_errors = {
        method: {
            label: 1
            - statistics.mean(
                row["metrics"]["per_class"][label]["recall"]
                for result in results
                for row in result["learning"]
                if row["budget_per_class"] == error_budget and row["method"] == method
            )
            for label in LABELS
        }
        for method in ("hdc", "logistic_regression", "mlp")
    }
    class_error_rows = "\n".join(
        f"| {PATTERNS[label].capitalize()} | "
        + " | ".join(percent(class_errors[method][label]) for method in class_errors)
        + " |"
        for label in LABELS
    )
    learning_rows, learning_headline, learning_takeaway, hardest_pattern, learned = (
        learning_comparison(summary, config, class_errors, error_budget)
    )
    fit_rows = "\n".join(
        "| "
        + str(budget)
        + " | "
        + " | ".join(
            f"{statistics.median(next(row for row in result['learning'] if row['budget_per_class'] == budget and row['method'] == method)['fit']['fit_ms'] for result in results):.2f}"
            for method in ("hdc", "logistic_regression", "mlp")
        )
        + " |"
        for budget in config.budgets
    )
    classifier_bytes = {
        method: statistics.median(
            next(
                row
                for row in result["learning"]
                if row["budget_per_class"] == max(config.budgets) and row["method"] == method
            )["fit"]["persistence"]["bytes"]
            for result in results
        )
        for method in ("logistic_regression", "mlp")
    }
    warning_count = sum(
        bool(row["fit"]["convergence_warnings"])
        for result in results
        for row in result["learning"]
        if row["method"] != "hdc"
    )
    frozen = read(run_dir / "frozen_settings.json")
    trial_fits = [info for trial in frozen["trials"] for info in trial["fit_diagnostics"]]
    selected_rows = "\n".join(
        f"| {budget} | {frozen['parameters'][str(budget)]['logistic_regression']['C']} | {frozen['parameters'][str(budget)]['logistic_regression']['scaling']} | {frozen['parameters'][str(budget)]['mlp']['width']} | {frozen['parameters'][str(budget)]['mlp']['alpha']} | {frozen['parameters'][str(budget)]['mlp']['scaling']} |"
        for budget in config.budgets
    )
    trial_warning_count = sum(bool(info["convergence_warnings"]) for info in trial_fits)
    predict = {
        method: timing_median(results, method + "_encode_predict") for method in UPDATE_METHODS
    }
    hdc_encode, hdc_score = (
        timing_median(results, "hdc_encode"),
        timing_median(results, "hdc_score"),
    )
    dimension = f"{config.dimension:,}"
    inputs = results[0]["resources"]["model_input_bytes"] // 4
    prediction_rows = "\n".join(
        [
            f"| HDC | Build the {dimension}-number hypervector ({ms(hdc_encode)} ms), then compare it with four class memories ({ms(hdc_score)} ms) | {ms(predict['hdc'])} |",
            f"| LR | Assemble {inputs} scaled measurements, then apply the trained linear model | {ms(predict['logistic_regression'])} |",
            f"| MLP | Assemble the same {inputs} inputs, then apply the trained one-hidden-layer network | {ms(predict['mlp'])} |",
        ]
    )
    prediction_finding = (
        f"HDC prediction takes {predict['hdc'] / predict['logistic_regression']:.1f}× as long as LR's. "
        f"That is expected: {100 * hdc_encode / predict['hdc']:.0f}% of HDC's time goes into building the hypervector, "
        f"binding, permuting and bundling every measured fact into {dimension} numbers. LR and the MLP read the same "
        f"joined measurements as {inputs} numbers, so preparing their input is little more than copying values. "
        f"All three predict in at most {ms(max(predict.values()))} ms, so prediction cost does not separate the methods."
    )
    reported = [
        (
            result["resources"]["timings"][method + "_encode_predict"][metric]
            for metric in ("median_ms", "p95_ms")
        )
        for result in results
        for method in UPDATE_METHODS
    ] + [
        (row["timings"]["complete_wall_ms"][metric] for metric in ("median_ms", "p95_ms"))
        for result in results
        for row in result["resources"]["learning_updates"]["rows"]
    ]
    p95_overhead = max(100 * (p95 / median - 1) for median, p95 in reported)
    update_single = min(update_batches(results))
    update_single_ms = {
        method: update_median(results, method, update_single) for method in UPDATE_METHODS
    }
    counts = results[0]["candidate_counts"]
    path_gain = summary["paired_differences"]["hdc_minus_hdc_without_paths"]
    order_gain = summary["paired_differences"]["hdc_minus_hdc_without_order"]
    effect_examples = {}
    for name, predicate in [
        ("helps", lambda row: row["net_correct"] > 0),
        ("hurts", lambda row: row["net_correct"] < 0),
        ("unchanged", lambda row: row["changed"] == 0),
    ]:
        found = next((row for row in online["effects"] if predicate(row)), None)
        effect_examples[name] = found
    save_json(root / "update_examples.json", effect_examples)
    effect_text = "\n".join(
        f"- **{name.capitalize()}:** "
        + (
            f"{row['episode_id']}: {row['corrected']} corrected, {row['regressed']} regressed, {row['changed']} "
            + ("prediction changed." if row["changed"] == 1 else "predictions changed.")
            if row
            else "No example occurred in the first fixed replay; none was manufactured."
        )
        for name, row in effect_examples.items()
    )
    mismatches = len(read(paths[0] / "retrieval_errors.json"))
    errors = read(paths[0] / "retrieval_errors.json")[:5]
    error_text = (
        "\n".join(
            f"| {row['episode_id']} | {PATTERNS[row['label']]} | {PATTERNS[row['retrieved_label']]} |"
            for row in errors
        )
        or "| — | No top-1 mismatch in this run | — |"
    )
    operator = results[0]["operators"]
    all_errors = read(paths[0] / "retrieval_errors.json")
    top_hits_checked = sum(r["evidence_audit"]["top_hits_checked"] for r in results)
    hardest = learned["hardest"]
    hardest_misses = sum(row["label"] == hardest for row in all_errors)
    retrieval = summary["retrieval"]["hdc"]
    precision, top1 = (
        percent(retrieval["precision_at_5"]["mean"]),
        percent(retrieval["top1"]["mean"]),
    )
    random_precision = percent(summary["random_precision_at_5"])
    quantization = statistics.mean(
        r["float16_quantization"]["precision_at_5_change"] for r in results
    )
    retrieval_headline = (
        f"HDC's first result has the query's pattern for {top1} of queries, and {precision} of its top five do, "
        f"against {random_precision} expected from random ranking. Storing the search vectors in float16 halves "
        f"their size with no measurable loss: precision@5 changes by {quantization * 100:.3f} points."
    )
    retrieval_miss_summary = (
        f"The first run has {mismatches} top-1 mismatches out of {results[0]['n_queries']} queries"
        + (
            f"; {hardest_misses} involve {PATTERNS[hardest]}, the same pattern HDC finds hardest to classify in Experiment 3."
            if hardest_misses * 2 > mismatches
            else "."
        )
        + " All are retained:"
        if mismatches
        else "The first run has no top-1 mismatches."
    )
    original, edited = case["original_top5"], case["without_handset_top5"]
    handset_edit_overlap = f"After the edit, {len(set(original) & set(edited))} of the original top five results remain in the top five."
    raw_bytes = results[0]["resources"]["hdc_raw_vector_bytes"]
    input_bytes = results[0]["resources"]["model_input_bytes"]
    storage_ratio = raw_bytes / input_bytes
    single = update_single_ms
    fits = {
        method: [
            statistics.median(
                next(
                    row
                    for row in result["learning"]
                    if row["budget_per_class"] == budget and row["method"] == method
                )["fit"]["fit_ms"]
                for result in results
            )
            for budget in sorted(config.budgets)
        ]
        for method in ("logistic_regression", "mlp")
    }
    few = (
        f"**Learns from very few reviews.** With {learned['first']} review per class, HDC scores "
        f"{min(learned['gaps']):.1f}–{max(learned['gaps']):.1f} points higher macro F1 than LR and the MLP."
        if learned["ahead"]
        else "**Learns from few reviews.** Experiment 3 compares macro F1 at every review budget."
    )
    strengths = [
        f"- {few}",
        "\n".join(
            [
                "- **Learning from a new review costs almost nothing, and the cost does not grow.**",
                f"  - **HDC:** {ms(single['hdc'])} ms to add a reviewed incident to memory: one vector addition, however many reviews came before.",
                f"  - **LR and the MLP:** {ms(single['logistic_regression'])} ms and {ms(single['mlp'])} ms to retrain on all retained reviews, every update.",
                f"  - **Retraining slows as reviews accumulate:** in our initial fits, from {fits['logistic_regression'][0]:.2f} to {fits['logistic_regression'][-1]:.2f} ms for LR and from {fits['mlp'][0]:.0f} to {fits['mlp'][-1]:.0f} ms for the MLP, between {min(config.budgets) * 4} and {max(config.budgets) * 4} reviews.",
                "  - **No retained history:** HDC needs no earlier reviews kept, and every update can be undone exactly.",
            ]
        ),
        (
            f"- **Finds comparable incidents and shows why.** Retrieval reaches {precision} precision@5, against {random_precision} "
            "for random ranking. Every similarity score breaks down exactly into contributions from individual facts, "
            "each traceable to its source records."
        ),
        (
            "- **One representation, many uses.** The same hypervector serves retrieval, classification, explanation and "
            "editing; a fact such as the handset can be removed from a query without re-encoding the rest."
        ),
    ]
    key_strengths = "\n".join(strengths)
    # Storage first: it is the cost every HDC user should expect.
    tradeoffs = [
        (
            f"- **More storage.** Each incident needs {raw_bytes:,} bytes as a hypervector, about {storage_ratio:.0f}× the "
            f"{input_bytes} bytes of its raw measurements."
        )
    ]
    if learned["on_par"]:
        tradeoffs.append(
            f"- **On par, not ahead, once reviews accumulate.** From {learned['on_par']} reviews per class, the three "
            f"methods are within {learned['spread']:.1f} points of one another."
        )
    tradeoffs += [
        "- **The class memory doesn't learn which facts matter.**",
        "  - It is a running sum of reviewed incidents, so each fact keeps the fixed weight the encoder gave it. LR and the MLP learn a weight for each input from the labels.",
        "  - A pattern decided by a few facts can be outvoted by facts that vary.",
        "  - This concerns the class memory used here, not HDC encoding; see [Future work](#future-work-teaching-the-class-memory-which-facts-matter).",
    ]
    key_tradeoffs = "\n".join(tradeoffs)
    claim_rows = [
        (
            "Composable representation",
            "Role swaps, reversed order and a rewired edge all change the encoding; without the relevant operator, the encodings are identical.",
            "Controlled pairs confirm the design; they do not show general graph reasoning.",
        ),
        (
            "Retrieval with explanations",
            f"{precision} precision@5 against {random_precision} for random ranking. All {top_hits_checked:,} top results trace to source records and reconstruct their scores exactly.",
            "No non-HDC retrieval baseline. Ablation gains partly reflect how the patterns are defined. Contributions are not causes.",
        ),
        (
            "Learning from few reviews",
            learning_takeaway,
            "The advantage shrinks to parity as reviews accumulate.",
        ),
        (
            "Updates without retraining",
            f"Fixed encoder; each delayed review is added in place and can be reversed exactly. {percent(online['metrics']['hdc']['accuracy'])} correct in the delayed-feedback replay.",
            "Still supervised: labels are required, and an individual update can make predictions worse.",
        ),
        (
            "Compute cost",
            f"HDC learns from one new review in {ms(update_single_ms['hdc'])} ms, versus {ms(update_single_ms['logistic_regression'])} ms (LR) and {ms(update_single_ms['mlp'])} ms (MLP) for full retraining. All three predict in at most {ms(max(predict.values()))} ms.",
            "One retraining run can absorb a whole batch, so large batches narrow or reverse the gap for LR. Incremental LR/MLP optimizers, energy and production serving were not measured.",
        ),
        (
            "Storage",
            f"{results[0]['resources']['hdc_raw_vector_bytes']:,} bytes per raw hypervector versus {results[0]['resources']['model_input_bytes']} bytes per LR/MLP input vector.",
            "HDC costs more storage here; the float16 search copy halves it without measurable loss.",
        ),
    ]
    claims = "\n".join(
        f"| {name} | {evidence} | {boundary} |" for name, evidence, boundary in claim_rows
    )
    report = render_template(
        "summary.md",
        {
            "review_delay_s": config.review_delay_s,
            "episode_count": f"{config.blocks * config.episodes_per_block:,}",
            "block_count": config.blocks,
            "memory_blocks": config.memory_blocks,
            "validation_blocks": config.validation_blocks,
            "test_blocks": config.blocks - config.memory_blocks - config.validation_blocks,
            "data_seed_count": len(config.data_seeds),
            "encoder_seed_count": len(config.encoder_seeds),
            "final_query_count": f"{summary['final_queries_scored']:,}",
            "final_world_count": summary["independent_final_worlds"],
            "corridor_segments": geography["corridor_segments"],
            "geography_attribution": geography["attribution"],
            "geography_dataset_url": geography["dataset_url"],
            "geography_licence_url": geography["licence_url"],
            "first_phone_signal": f"{case['query']['observations'][0]['radio_dbm']:.1f}",
            "last_phone_signal": f"{case['query']['observations'][2]['radio_dbm']:.1f}",
            "middle_link_loss": f"{case['query']['observations'][1]['link_loss_pct']:.1f}",
            "middle_peer_loss": f"{case['query']['observations'][1]['peer_loss_pct']:.1f}",
            "query_episode_id": case["query"]["episode_id"],
            "query_pattern": PATTERNS[case["query_label"]],
            "candidate_episode_id": case["candidate"]["episode_id"],
            "candidate_pattern": PATTERNS[case["candidate_label"]],
            "encoder_explanation": encoder_explanation(config),
            "phone_signal_weight": f"{config.local_weight:g}",
            "context_weight": f"{config.context_weight:g}",
            "handset_weight": f"{config.handset_weight:g}",
            "dimension": f"{config.dimension:,}",
            "bound_cosine": f"{operator['role_binding']['bound_cosine']:.3f}",
            "unbound_error": f"{operator['role_binding']['without_binding_max_error']:.2g}",
            "ordered_cosine": f"{operator['event_order']['with_order_cosine']:.3f}",
            "unordered_error": f"{operator['event_order']['without_order_max_error']:.2g}",
            "connected_cosine": f"{operator['connectivity']['with_path_cosine']:.3f}",
            "disconnected_error": f"{operator['connectivity']['without_path_max_error']:.2g}",
            "adjacent_level_cosine": f"{operator['numeric_neighbourhood']['adjacent_cosine']:.3f}",
            "distant_level_cosine": f"{operator['numeric_neighbourhood']['far_cosine']:.3f}",
            "handset_edit_error": f"{case['query_edit_max_error']:.2g}",
            "original_top_five": ", ".join(case["original_top5"]),
            "edited_top_five": ", ".join(case["without_handset_top5"]),
            "minimum_candidates": counts["min"],
            "maximum_candidates": counts["max"],
            "mean_candidates": f"{counts['mean']:.1f}",
            "retrieval_rows": retrieval_rows,
            "random_precision": percent(summary["random_precision_at_5"]),
            "path_gain_mean": f"{path_gain['mean'] * 100:.1f}",
            "path_gain_lower": f"{path_gain['lower'] * 100:.1f}",
            "path_gain_upper": f"{path_gain['upper'] * 100:.1f}",
            "order_gain_mean": f"{order_gain['mean'] * 100:.1f}",
            "order_gain_lower": f"{order_gain['lower'] * 100:.1f}",
            "order_gain_upper": f"{order_gain['upper'] * 100:.1f}",
            "reconstruction_error": f"{max(r['evidence_audit']['max_raw_reconstruction_error'] for r in results):.2g}",
            "score_reconstruction_error": f"{max(r['evidence_audit']['max_score_reconstruction_error'] for r in results):.2g}",
            "retrieval_mismatches": mismatches,
            "query_count": results[0]["n_queries"],
            "retrieval_error_rows": error_text,
            "quantization_precision_change": f"{statistics.mean(r['float16_quantization']['precision_at_5_change'] for r in results) * 100:.3f}",
            "learning_rows": learning_rows,
            "key_strengths": key_strengths,
            "key_tradeoffs": key_tradeoffs,
            "retrieval_headline": retrieval_headline,
            "retrieval_miss_summary": retrieval_miss_summary,
            "handset_edit_overlap": handset_edit_overlap,
            "top_hits_checked": f"{top_hits_checked:,}",
            "learning_headline": learning_headline,
            "learning_takeaway": learning_takeaway,
            "hardest_pattern": hardest_pattern,
            "delayed_accuracy": percent(online["metrics"]["hdc"]["accuracy"]),
            "delayed_coverage": percent(online["metrics"]["hdc"]["coverage"]),
            "error_budget": error_budget,
            "classifier_warning_count": warning_count,
            "classifier_fit_count": len(results) * len(config.budgets) * 2,
            "selection_warning_count": trial_warning_count,
            "selection_fit_count": len(trial_fits),
            "class_error_rows": class_error_rows,
            "update_effects": effect_text,
            "pair_count": len(results),
            "p95_overhead": f"{p95_overhead:.0f}",
            "runtime_platform": manifest["runtime"]["platform"],
            "runtime_python": manifest["runtime"]["python"],
            "prediction_rows": prediction_rows,
            "prediction_finding": prediction_finding,
            "raw_vector_bytes": f"{results[0]['resources']['hdc_raw_vector_bytes']:,}",
            "search_vector_bytes": f"{results[0]['resources']['hdc_search_vector_bytes']:,}",
            "model_input_bytes": results[0]["resources"]["model_input_bytes"],
            "class_memory_bytes": f"{results[0]['resources']['class_memory_bytes']['hdc']:,}",
            "representation_mib": f"{results[0]['resources']['hdc_representation_tensor_bytes'] / 2**20:.2f}",
            "model_input_mib": f"{results[0]['resources']['model_input_tensor_bytes'] / 2**20:.2f}",
            "encoder_cache_mib": f"{results[0]['resources']['encoder_cached_basis_bytes'] / 2**20:.2f}",
            "vector_store_mib": f"{results[0]['resources']['vector_store_bytes'] / 2**20:.2f}",
            "source_store_mib": f"{results[0]['resources']['source_store_bytes'] / 2**20:.2f}",
            "max_reviews_per_class": max(config.budgets),
            "lr_model_bytes": f"{classifier_bytes['logistic_regression']:,.0f}",
            "mlp_model_bytes": f"{classifier_bytes['mlp']:,.0f}",
            "review_buffer_bytes": f"{results[0]['resources']['review_buffer_feature_and_label_bytes']:,}",
            "fit_rows": fit_rows,
            "encode_seconds": f"{results[0]['resources']['encode_all_s']:.3f}",
            "persistence_seconds": f"{results[0]['resources']['persistence_s']:.3f}",
            "eligibility_seconds": f"{results[0]['resources']['exact_gate_s']:.3f}",
            "lance_retrieval_ms": f"{results[0]['resources']['retrieval_latency']['hdc_lance_float16']['median_ms']:.3f}",
            "memory_retrieval_ms": f"{results[0]['resources']['retrieval_latency']['hdc']['median_ms']:.3f}",
            "learning_update_explanation": learning_update_explanation(results, config),
            "claims_rows": claims,
            "geographic_enhancements": geographic_enhancement_explanation(),
            "future_work": future_work_explanation(
                run_dir,
                config,
                paths[0],
                config.data_seeds[0],
                config.encoder_seeds[0],
                error_budget,
                learned["hardest"],
            ),
            "source_link": source_link,
            "data_seeds": ", ".join(str(seed) for seed in config.data_seeds),
            "selected_settings_rows": selected_rows,
        },
    )
    (root / "summary.md").write_text(report)
    save_json(
        root / "claims.json",
        {
            "code_hash": manifest["run_code_hash"],
            "report_generator_hash": file_hash(Path(__file__)),
            "report_template_hashes": {
                path.name: file_hash(path) for path in sorted(REPORT_TEMPLATES.glob("*.md"))
            },
            "summary_hash": manifest["summary_hash"],
            "figure_hashes": {
                str(path.relative_to(root)): file_hash(path)
                for path in sorted((root / "figures").iterdir())
                if path.is_file()
            },
            "claims": [{"claim": a, "evidence": b, "boundary": c} for a, b, c in claim_rows],
            "report_hash": file_hash(root / "summary.md"),
        },
    )
    print(f"Report written to {root / 'summary.md'}", flush=True)
    return root / "summary.md"
