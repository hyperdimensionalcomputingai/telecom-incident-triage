"""Readable evidence, exportable figures, and measured claim boundaries."""

import json
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

    methods = (
        "hdc_encode_predict",
        "logistic_regression_encode_predict",
        "mlp_encode_predict",
        "hdc_encode_score_update",
        "logistic_regression_review_refit",
        "mlp_review_refit",
    )
    names = (
        "HDC encode + predict",
        "LR encode + predict",
        "MLP encode + predict",
        "HDC encode + predict + add review",
        "LR incorporate review by batch refit",
        "MLP incorporate review by batch refit",
    )
    medians = [
        statistics.median(result["resources"]["timings"][method]["median_ms"] for result in results)
        for method in methods
    ]
    p95 = [
        statistics.median(result["resources"]["timings"][method]["p95_ms"] for result in results)
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(10, 5.6), layout="constrained")
    axis.barh(range(6), medians, color=["#287b78", "#a06935", "#7461a2"] * 2, height=0.6)
    axis.scatter(
        p95, range(6), color="#263647", marker="|", s=170, label="Median of run p95 measurements"
    )
    axis.set_yticks(range(6), names)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Elapsed time in milliseconds, logarithmic scale; database work excluded")
    axis.set_title("Prediction and incorporation of one further review", loc="left")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21))
    save_figure(figure, root, "compute", plt)

    batches = sorted(
        {
            row["batch_size"]
            for result in results
            for row in result["resources"]["learning_updates"]["rows"]
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout="constrained")
    update_names = {
        "hdc": "HDC additive memory",
        "logistic_regression": "LR batch refit",
        "mlp": "MLP batch refit",
    }
    for method in ("hdc", "logistic_regression", "mlp"):
        totals = [
            statistics.median(
                row["timings"]["complete_wall_ms"]["median_ms"]
                for result in results
                for row in result["resources"]["learning_updates"]["rows"]
                if row["method"] == method and row["batch_size"] == batch
            )
            for batch in batches
        ]
        for axis, values in zip(
            axes, (totals, [total / batch for total, batch in zip(totals, batches)])
        ):
            axis.plot(
                batches,
                values,
                "-o",
                linewidth=2.4,
                color=COLOURS[method],
                label=update_names[method],
            )
            axis.set_xticks(batches)
            axis.set_xlabel("Newly reviewed incidents in the arriving batch")
            axis.set_yscale("log")
    axes[0].set_ylabel("Complete batch wall latency, ms (log scale)")
    axes[1].set_ylabel("Amortized wall latency per new review, ms (log scale)")
    axes[0].set_title("Encode, incorporate reviews, then predict", loc="left")
    axes[1].set_title("Batch refit cost shared across new reviews", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    initial = results[0]["resources"]["learning_updates"]["rows"][0]["initial_training_examples"]
    figure.suptitle(
        f"Same {initial} starting reviews; fixed encoder and frozen LR/MLP settings", fontsize=13
    )
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
    batches = sorted(
        {
            row["batch_size"]
            for result in results
            for row in result["resources"]["learning_updates"]["rows"]
        }
    )
    rows, stages = [], []
    for batch in batches:
        for method in ("hdc", "logistic_regression", "mlp"):
            trials = [
                row
                for result in results
                for row in result["resources"]["learning_updates"]["rows"]
                if row["batch_size"] == batch and row["method"] == method
            ]

            def median(group, key, metric="median_ms", trials=trials):
                return statistics.median(row[group][key][metric] for row in trials)

            name = {
                "hdc": "HDC additive class memory",
                "logistic_regression": "LR full batch refit",
                "mlp": "MLP full batch refit",
            }[method]
            rows.append(
                f"| {batch} | {name} | {trials[0]['total_reviewed_examples']} | "
                f"{median('timings', 'complete_wall_ms'):.3f} | {median('timings', 'complete_wall_ms', 'p95_ms'):.3f} | "
                f"{median('per_new_sample', 'complete_wall_ms'):.4f} | "
                f"{median('per_new_sample', 'complete_cpu_ms'):.4f} |"
            )
            stages.append(
                f"| {batch} | {name} | "
                + " | ".join(
                    f"{median('per_new_sample', key):.4f}"
                    for key in ("encoding_ms", "learning_ms", "prediction_ms")
                )
                + " |"
            )
    initial = results[0]["resources"]["learning_updates"]["rows"][0]["initial_training_examples"]
    largest_batch = max(batches)
    costs = {
        method: statistics.median(
            row["per_new_sample"]["complete_wall_ms"]["median_ms"]
            for result in results
            for row in result["resources"]["learning_updates"]["rows"]
            if row["method"] == method and row["batch_size"] == largest_batch
        )
        for method in ("hdc", "logistic_regression", "mlp")
    }
    comparison = (
        "LR is cheaper per new review than HDC at this batch size."
        if costs["logistic_regression"] < costs["hdc"]
        else "HDC is cheaper per new review than LR at this batch size."
    )
    finding = (
        f"At {largest_batch} new reviews, complete amortized wall cost is "
        f"{costs['hdc']:.4f} ms per review for HDC, {costs['logistic_regression']:.4f} ms for LR, "
        f"and {costs['mlp']:.4f} ms for the MLP. {comparison} "
        "The batch size changes the practical cost comparison; a one-review refit cost should not be extrapolated by multiplying it by the number of arriving reviews."
    )
    warning_count = sum(
        bool(info["convergence_warnings"])
        for result in results
        for row in result["resources"]["learning_updates"]["rows"]
        for info in row["fit_diagnostics"]
    )
    table = "\n".join(rows)
    breakdown = "\n".join(stages)
    skipped = results[0]["resources"]["learning_updates"]["skipped_batches"]
    skipped_text = (
        "The small run omits batches without enough later memory reviews: "
        + ", ".join(str(row["batch_size"]) for row in skipped)
        + "."
        if skipped
        else "Every requested batch size was measured."
    )
    return render_template(
        "learning-updates.md",
        {
            "initial_reviews": initial,
            "max_reviews_per_class": max(config.budgets),
            "batch_sizes": ", ".join(str(batch) for batch in batches),
            "reviews_after_twenty": initial + 20,
            "reviews_after_forty": initial + 40,
            "refit_warmup": config.refit_warmup,
            "refit_repeats": config.refit_repeats,
            "batch_coverage": skipped_text,
            "learning_update_rows": table,
            "learning_update_stage_rows": breakdown,
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
    learning_rows = "\n".join(
        "| "
        + str(budget)
        + " | "
        + " | ".join(
            interval_text(summary["learning"][str(budget)][method])
            for method in ("hdc", "logistic_regression", "mlp")
        )
        + " |"
        for budget in config.budgets
    )
    error_budget = 5 if 5 in config.budgets else min(config.budgets)
    class_error_rows = "\n".join(
        "| "
        + PATTERNS[label].capitalize()
        + " | "
        + " | ".join(
            percent(
                1
                - statistics.mean(
                    row["metrics"]["per_class"][label]["recall"]
                    for result in results
                    for row in result["learning"]
                    if row["budget_per_class"] == error_budget and row["method"] == method
                )
            )
            for method in ("hdc", "logistic_regression", "mlp")
        )
        + " |"
        for label in LABELS
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
    learning_difference = summary["learning_differences"][str(error_budget)]
    difference_text = "; ".join(
        f"HDC minus {name}: {learning_difference['hdc_minus_' + method]['mean'] * 100:.1f} percentage points [{learning_difference['hdc_minus_' + method]['lower'] * 100:.1f}, {learning_difference['hdc_minus_' + method]['upper'] * 100:.1f}]"
        for method, name in [("logistic_regression", "LR"), ("mlp", "MLP")]
    )
    timing_rows = "\n".join(
        f"| {name} | {statistics.median(result['resources']['timings'][key]['median_ms'] for result in results):.4f} ms | {statistics.median(result['resources']['timings'][key]['p95_ms'] for result in results):.4f} ms |"
        for key, name in [
            ("hdc_add_only", "HDC addition only"),
            ("hdc_encode", "HDC encoding"),
            ("hdc_score", "HDC scoring"),
            ("hdc_encode_predict", "HDC encode–predict"),
            ("logistic_regression_encode_predict", "LR encode–predict"),
            ("mlp_encode_predict", "MLP encode–predict"),
            ("hdc_encode_score_update", "HDC encode–score–update"),
            ("logistic_regression_review_refit", "LR encode–refit–predict"),
            ("mlp_review_refit", "MLP encode–refit–predict"),
        ]
    )
    counts = results[0]["candidate_counts"]
    path_gain = summary["paired_differences"]["hdc_minus_hdc_without_paths"]
    order_gain = summary["paired_differences"]["hdc_minus_hdc_without_order"]
    best_budget = max(config.budgets)
    last_f1 = summary["learning"][str(best_budget)]["hdc"]["mean"]
    learning_direction = (
        "More reviews improve the aggregate result over the one-example starting point."
        if last_f1 > summary["learning"][str(min(config.budgets))]["hdc"]["mean"]
        else "More reviews do not improve the aggregate result over the one-example starting point."
    )
    budget_80 = next(
        (
            budget
            for budget in config.budgets
            if summary["learning"][str(budget)]["hdc"]["mean"] >= 0.8
        ),
        None,
    )
    low_sample = (
        f"HDC reaches mean macro F1 ≥0.80 with {budget_80} "
        + ("review" if budget_80 == 1 else "reviews")
        + f" per class ({budget_80 * 4} total)."
        if budget_80
        else f"HDC does not reach mean macro F1 0.80 within {best_budget} reviews per class."
    )
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
    claim_rows = [
        (
            "Composable representation",
            "Role swaps, order changes and edge rewiring are detectable; controls without the relevant operator remain invariant.",
            "Controlled illustrations and ablations support the designed representation, not arbitrary graph reasoning.",
        ),
        (
            "Inspectable evidence",
            f"All {sum(r['evidence_audit']['top_hits_checked'] for r in results):,} retrieved top hits resolve to source records and reconstruct their float32 reference scores.",
            "Provenance is retained alongside vectors; arithmetic contributions include interference and do not establish causes. Float16 search has a separately recorded quantization residual.",
        ),
        (
            "Online learning",
            f"Frozen encoder, delayed reviews, additive class memories, exact reversal; {low_sample}",
            "Supervised learning still requires labels. Updates can regress, and the scenarios are simulated.",
        ),
        (
            "Compute cost",
            "Addition, prediction and review incorporation are measured separately on CPU against trained LR and MLP.",
            "Fast addition alone is not serving latency, energy efficiency, or an advantage over the measured controls.",
        ),
        (
            "Storage",
            f"{results[0]['resources']['hdc_raw_vector_bytes']:,} bytes per raw hypervector versus {results[0]['resources']['model_input_bytes']} bytes per LR/MLP input vector.",
            "This study shows no storage saving from HDC.",
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
            "learning_direction": learning_direction,
            "low_sample_finding": low_sample,
            "delayed_accuracy": percent(online["metrics"]["hdc"]["accuracy"]),
            "delayed_coverage": percent(online["metrics"]["hdc"]["coverage"]),
            "error_budget": error_budget,
            "learning_difference": difference_text,
            "classifier_warning_count": warning_count,
            "classifier_fit_count": len(results) * len(config.budgets) * 2,
            "selection_warning_count": trial_warning_count,
            "selection_fit_count": len(trial_fits),
            "class_error_rows": class_error_rows,
            "update_effects": effect_text,
            "warmup": config.warmup,
            "benchmark_repeats": config.benchmark_repeats,
            "refit_warmup": config.refit_warmup,
            "refit_repeats": config.refit_repeats,
            "runtime_platform": manifest["runtime"]["platform"],
            "runtime_python": manifest["runtime"]["python"],
            "timing_rows": timing_rows,
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
            "refit_training_examples": results[0]["resources"]["review_refit_training_examples"],
            "learning_update_explanation": learning_update_explanation(results, config),
            "claims_rows": claims,
            "geographic_enhancements": geographic_enhancement_explanation(),
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
