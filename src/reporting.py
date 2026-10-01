"""Readable evidence, exportable figures, and measured claim boundaries."""
from pathlib import Path
import json
import statistics
import textwrap
import os
import math
from config import LABELS, ROOT, Config, digest, file_hash, save_json

COLOURS = {'hdc': '#287b78', 'logistic_regression': '#a06935', 'mlp': '#7461a2',
           'structured': '#a06935', 'hdc_without_paths': '#929eaa', 'hdc_without_order': '#7461a2'}
NAMES = {'hdc': 'HDC', 'structured': 'Explicit structured comparator',
         'hdc_without_paths': 'HDC: connectivity omitted', 'hdc_without_order': 'HDC: order omitted',
         'hdc_lance_float16': 'HDC in Lance (float16)', 'logistic_regression': 'Trained logistic regression',
         'mlp': 'Small trained MLP'}
PATTERNS = {'radio_deteriorating': 'deteriorating radio', 'transient_recovery': 'transient disruption and recovery',
            'shared_transport': 'shared transport impairment', 'normal': 'normal service'}


def read(path):
    return json.loads(Path(path).read_text())


def percent(value):
    return f'{value * 100:.1f}%'


def interval_text(value):
    return f'{percent(value["mean"])} [{percent(value["lower"])}, {percent(value["upper"])}]'


def figure_style():
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/telus-matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/telus-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 14,
        'axes.labelcolor': '#344454', 'text.color': '#263647', 'axes.edgecolor': '#c3cbd2',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': '#f4f6f8',
        'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': 'white', 'grid.linewidth': 1.2,
        'axes.axisbelow': True, 'savefig.facecolor': 'white'})
    return plt


def save_figure(figure, root, name, plt):
    figure.savefig(root / f'{name}.png', dpi=180, bbox_inches='tight')
    figure.savefig(root / f'{name}.svg', bbox_inches='tight')
    plt.close(figure)


def make_schema_figure(root, plt):
    """A conceptual projection of the joined records, with illustrative values."""
    import io
    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

    background, ink, muted, line, orange = '#FAF9F6', '#15141A', '#5A5864', '#E2DFD8', '#CD6F3E'
    figure, axis = plt.subplots(figsize=(16, 10.8))
    figure.patch.set_facecolor(background)
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.set(xlim=(0, 1600), ylim=(1080, 0))
    axis.set_axis_off()
    axis.text(55, 55, 'From one phone to a shared network dependency', fontsize=23, weight='bold', color=ink)
    axis.text(55, 92, 'Conceptual graph schema · example values at observation time t', fontsize=14, color=muted)

    def properties(x, y, entries, width=224):
        height = 18 + 25 * len(entries)
        axis.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0,rounding_size=6', facecolor='white', edgecolor=line, linewidth=1.2, zorder=4))
        axis.text(x + 12, y + 10, '\n'.join(entries), va='top', fontsize=11.5,
            fontfamily='DejaVu Sans Mono', linespacing=1.45, color=muted, zorder=5)

    def node(x, y, name, entries, active=False):
        axis.add_patch(Circle((x, y), 77, facecolor=background,
            edgecolor=orange if active else ink, linewidth=2, zorder=3))
        axis.text(x, y, name, ha='center', va='center', fontsize=14, weight='bold', color=ink, zorder=4)
        # A short connector ties each property box to the node's bottom-right corner.
        axis.plot([x + 50, x + 50], [y + 59, y + 95], color=line, linewidth=1.2, zorder=2)
        properties(x + 18, y + 95, entries)

    def edge(start, end, label, label_x, label_y, entries, active=False):
        colour = orange if active else muted
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
            linewidth=2.2 if active else 1.7, color=colour, shrinkA=0, shrinkB=2, zorder=1))
        axis.text(label_x, label_y, label, ha='center', va='bottom', fontsize=12.5,
            weight='bold', color=colour, zorder=5)
        properties(label_x - 98, label_y + 10, entries, width=196)

    # Orange identifies the selected commuter path; arrows denote relationships, not causes.
    edge((187, 260), (393, 260), 'HAS_PHONE', 290, 175, ['subscriber_id: S7'])
    edge((547, 260), (773, 260), 'SERVED_BY', 660, 175, ['timestamp_s: t'], active=True)
    edge((927, 260), (1173, 260), 'USES', 1050, 150,
         ['valid_from_s: t0', 'valid_to_s: t1'], active=True)
    edge((187, 780), (393, 780), 'HAS_OBSERVATION', 290, 695, ['ordinal: 1'])
    edge((470, 703), (470, 337), 'OF_PHONE', 340, 510, ['phone_id: P7'])
    edge((927, 780), (1173, 780), 'AT_CELL', 1050, 695, ['cell_id: C5'])
    edge((1250, 703), (1250, 337), 'USES', 1370, 510,
         ['valid_from_s: t0', 'valid_to_s: t1'])

    node(110, 260, 'Subscriber', ['subscriber_id: S7', 'pseudonym: Traveller7'])
    node(470, 260, 'Phone', ['phone_id: P7', 'handset: model_1'], active=True)
    node(850, 260, 'Serving\ncell', ['cell_id: C2', 'load_pct(t): 62'], active=True)
    node(1250, 260, 'Backhaul\nlink', ['link_id: L0', 'loss_pct(t): 4.8', 'latency_ms(t): 60'], active=True)
    node(110, 780, 'Episode', ['episode_id: E7', 'observations: 3'])
    node(470, 780, 'Observation', ['observation_id: O1', 'timestamp_s: t', 'radio_dbm: -104', 'geometry: Point(x,y)'])
    node(850, 780, 'Peer\nobservation', ['peer_obs_id: PO1', 'timestamp_s: t', 'peer_radio_dbm: -82', 'peer_loss_pct: 4.7'])
    node(1250, 780, 'Peer\'s\ncell', ['cell_id: C5'])

    axis.text(60, 1020, 'Orange: commuter dependency path', fontsize=13, color=orange, weight='bold')
    axis.text(510, 1020, 'Both cells use L0 at t: peers can share an upstream problem.', fontsize=13, color=ink)
    axis.text(60, 1053, 'Circles are nodes; arrows are edges. Property boxes contain illustrative key-value pairs.', fontsize=12, color=muted)
    # Keep HTML as the editable display source; SVG and PNG are matching figure exports.
    with plt.rc_context({'svg.fonttype': 'none'}):
        output = io.StringIO()
        figure.savefig(output, format='svg', facecolor=background)
    svg = output.getvalue()
    svg = svg.replace("font-family: 'DejaVu Sans';", "font-family: 'DejaVu Sans', Arial, sans-serif;")
    svg = svg.replace("font-family: 'DejaVu Sans Mono';", "font-family: 'DejaVu Sans Mono', 'Courier New', monospace;")
    svg = svg.replace('<svg ', '<svg role="img" aria-labelledby="telecom-schema-title telecom-schema-desc" ', 1)
    start = svg.index('>', svg.index('<svg ')) + 1
    svg = svg[:start] + '''
 <title id="telecom-schema-title">Conceptual telecom incident graph schema</title>
 <desc id="telecom-schema-desc">A subscriber has a phone. An episode contains ordered observations of that phone.
 At time t the phone is served by cell C2, which uses backhaul link L0. A peer observation at cell C5
 shares L0 through another time-valid edge. Node property boxes sit at the bottom right; edge property
 boxes sit below the edge labels. Cell and link measurements are joined telemetry at t.</desc>''' + svg[start:]
    html = '<!doctype html><html lang="en-CA"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Telecom incident graph schema</title><style>body{margin:0;background:#FAF9F6}.figure{overflow:auto}svg{display:block;width:100%;min-width:1100px;height:auto}</style><div class="figure">' + svg[svg.index('<svg '):] + '</div></html>'
    (root / 'schema.html').write_text(html)
    (root / 'schema.svg').write_text(svg)
    figure.savefig(root / 'schema.png', dpi=180, facecolor=background)
    plt.close(figure)


def make_figures(root, summary, results, case, config, geography_root):
    plt = figure_style()
    root.mkdir(parents=True, exist_ok=True)
    make_schema_figure(root, plt)
    figure, axis = plt.subplots(figsize=(9, 4.8), layout='constrained')
    methods = ['hdc', 'structured', 'hdc_without_paths', 'hdc_without_order']
    points = [summary['retrieval'][method]['precision_at_5'] for method in methods]
    means = [point['mean'] for point in points]
    axis.barh(range(4), means, color=[COLOURS[method] for method in methods], height=.62)
    axis.errorbar(means, range(4), xerr=[[point['mean'] - point['lower'] for point in points],
        [point['upper'] - point['mean'] for point in points]], fmt='none', ecolor='#243342', capsize=4)
    axis.set_yticks(range(4), [NAMES[method] for method in methods])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.06)
    axis.set_xlabel('Share of the first five retrieved incidents with the same triage pattern')
    axis.set_title('Order and connected context improve incident retrieval', loc='left')
    for index, mean in enumerate(means):
        axis.text(min(mean + .018, 1.01), index, percent(mean), va='center', fontsize=10)
    axis.axvline(summary['random_precision_at_5'], color='#637181', linestyle='--', label='Random ranking reference')
    axis.legend(loc='lower right', frameon=False)
    save_figure(figure, root, 'retrieval', plt)

    figure, axis = plt.subplots(figsize=(8.6, 5), layout='constrained')
    for method in ('hdc', 'logistic_regression', 'mlp'):
        rows = [summary['learning'][str(budget)][method] for budget in config.budgets]
        means = [row['mean'] for row in rows]
        axis.plot(config.budgets, means, marker='o', linewidth=2.4, color=COLOURS[method], label=NAMES[method])
        axis.fill_between(config.budgets, [row['lower'] for row in rows], [row['upper'] for row in rows], color=COLOURS[method], alpha=.12)
    axis.set_xscale('log')
    axis.set_xticks(config.budgets, [str(budget) for budget in config.budgets])
    from matplotlib.ticker import NullFormatter, MaxNLocator
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(0, 1.05)
    axis.set_xlabel('Reviewed examples per class (four classes)')
    axis.set_ylabel('Macro F1 on independent final-test worlds')
    axis.set_title('Memory starts empty and grows through reviews', loc='left')
    axis.legend(frameon=False, loc='lower right')
    save_figure(figure, root, 'learning', plt)

    methods = ('hdc_encode_predict', 'logistic_regression_encode_predict', 'mlp_encode_predict', 'hdc_encode_score_update', 'logistic_regression_review_refit', 'mlp_review_refit')
    names = ('HDC encode + predict', 'LR encode + predict', 'MLP encode + predict', 'HDC encode + predict + add review', 'LR incorporate review by batch refit', 'MLP incorporate review by batch refit')
    medians = [statistics.median(result['resources']['timings'][method]['median_ms'] for result in results) for method in methods]
    p95 = [statistics.median(result['resources']['timings'][method]['p95_ms'] for result in results) for method in methods]
    figure, axis = plt.subplots(figsize=(10, 5.6), layout='constrained')
    axis.barh(range(6), medians, color=['#287b78', '#a06935', '#7461a2'] * 2, height=.6)
    axis.scatter(p95, range(6), color='#263647', marker='|', s=170, label='Median of run p95 measurements')
    axis.set_yticks(range(6), names)
    axis.invert_yaxis()
    axis.set_xscale('log')
    axis.set_xlabel('Elapsed time in milliseconds, logarithmic scale; database work excluded')
    axis.set_title('Prediction and incorporation of one further review', loc='left')
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.21))
    save_figure(figure, root, 'compute', plt)

    batches = sorted({row['batch_size'] for result in results
                      for row in result['resources']['learning_updates']['rows']})
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout='constrained')
    update_names = {'hdc': 'HDC additive memory', 'logistic_regression': 'LR batch refit', 'mlp': 'MLP batch refit'}
    for method in ('hdc', 'logistic_regression', 'mlp'):
        totals = [statistics.median(row['timings']['complete_wall_ms']['median_ms']
            for result in results for row in result['resources']['learning_updates']['rows']
            if row['method'] == method and row['batch_size'] == batch) for batch in batches]
        for axis, values in zip(axes, (totals, [total / batch for total, batch in zip(totals, batches)])):
            axis.plot(batches, values, '-o', linewidth=2.4, color=COLOURS[method], label=update_names[method])
            axis.set_xticks(batches)
            axis.set_xlabel('Newly reviewed incidents in the arriving batch')
            axis.set_yscale('log')
    axes[0].set_ylabel('Complete batch wall latency, ms (log scale)')
    axes[1].set_ylabel('Amortized wall latency per new review, ms (log scale)')
    axes[0].set_title('Encode, incorporate reviews, then predict', loc='left')
    axes[1].set_title('Batch refit cost shared across new reviews', loc='left')
    axes[0].legend(frameon=False, fontsize=9)
    initial = results[0]['resources']['learning_updates']['rows'][0]['initial_training_examples']
    figure.suptitle(f'Same {initial} starting reviews; fixed encoder and frozen LR/MLP settings', fontsize=13)
    save_figure(figure, root, 'learning-updates', plt)

    from geography import load_geography
    lines, _ = load_geography(geography_root)
    query = case['query']
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout='constrained')
    axis = axes[0]
    for line in lines:
        axis.plot([p[0] for p in line], [p[1] for p in line], color='#b9c1c8', linewidth=.85)
    observations = query['observations']
    xs, ys = [row['longitude'] for row in observations], [row['latitude'] for row in observations]
    axis.plot(xs, ys, color='#287b78', marker='o', markersize=6, linewidth=2)
    axis.set_xlim(min(xs) - .007, max(xs) + .007)
    axis.set_ylim(min(ys) - .003, max(ys) + .003)
    axis.set_title('1. A simulated commute', loc='left')
    axis.set_xlabel('Longitude (public road geometry)')
    axis.set_ylabel('Latitude')
    axis.ticklabel_format(useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis = axes[1]
    axis.plot([0, 20, 40], [row['radio_dbm'] for row in observations], '-o', color='#287b78', label='Received signal strength')
    axis.set_ylim(-125, -65)
    axis.set_xticks([0, 20, 40], ['Before', 'During', 'After'])
    axis.set_ylabel('Received signal strength (dBm)')
    axis.set_title('2. Three observations', loc='left')
    axis = axes[2]
    mid = observations[1]
    axis.grid(False)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect('equal')
    from matplotlib.patches import Circle
    nodes = [(0.17, .8, 'Phone'), (.73, .8, 'Serving\ncell'), (.73, .36, 'Backhaul\nlink'), (.17, .36, 'Peer\nphones')]
    for x, y, text in nodes:
        axis.add_patch(Circle((x, y), .135, facecolor='#e3efed', edgecolor='#639b97', linewidth=1.5))
        axis.text(x, y, text, ha='center', va='center', fontsize=9)
    for a, b in [((.305, .8), (.595, .8)), ((.73, .665), (.73, .495)), ((.305, .36), (.595, .36))]:
        axis.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': '#344454', 'lw': 1.6})
    axis.text(.97, .56, 'time-valid\nedge', ha='right', fontsize=9)
    axis.text(.5, .08, f'Link loss: {mid["link_loss_pct"]:.1f}%\nPeer loss: {mid["peer_loss_pct"]:.1f}%', ha='center', fontsize=10)
    axis.set_title('3. Inspect connected facts', loc='left')
    figure.suptitle('Public streets; every phone, network edge and measurement is simulated', fontsize=13)
    save_figure(figure, root, 'incident', plt)


def learning_update_explanation(results, config):
    batches = sorted({row['batch_size'] for result in results
                      for row in result['resources']['learning_updates']['rows']})
    rows, stages = [], []
    for batch in batches:
        for method in ('hdc', 'logistic_regression', 'mlp'):
            trials = [row for result in results for row in result['resources']['learning_updates']['rows']
                      if row['batch_size'] == batch and row['method'] == method]
            median = lambda group, key, metric='median_ms': statistics.median(row[group][key][metric] for row in trials)
            name = {'hdc': 'HDC additive class memory', 'logistic_regression': 'LR full batch refit',
                    'mlp': 'MLP full batch refit'}[method]
            rows.append(f'| {batch} | {name} | {trials[0]["total_reviewed_examples"]} | '
                f'{median("timings", "complete_wall_ms"):.3f} | {median("timings", "complete_wall_ms", "p95_ms"):.3f} | '
                f'{median("per_new_sample", "complete_wall_ms"):.4f} | '
                f'{median("per_new_sample", "complete_cpu_ms"):.4f} |')
            stages.append(f'| {batch} | {name} | ' + ' | '.join(
                f'{median("per_new_sample", key):.4f}'
                for key in ('encoding_ms', 'learning_ms', 'prediction_ms')) + ' |')
    initial = results[0]['resources']['learning_updates']['rows'][0]['initial_training_examples']
    largest_batch = max(batches)
    costs = {method: statistics.median(row['per_new_sample']['complete_wall_ms']['median_ms']
        for result in results for row in result['resources']['learning_updates']['rows']
        if row['method'] == method and row['batch_size'] == largest_batch)
        for method in ('hdc', 'logistic_regression', 'mlp')}
    comparison = ('LR is cheaper per new review than HDC at this batch size.'
                  if costs['logistic_regression'] < costs['hdc'] else
                  'HDC is cheaper per new review than LR at this batch size.')
    finding = (f'At {largest_batch} new reviews, complete amortized wall cost is '
        f'{costs["hdc"]:.4f} ms per review for HDC, {costs["logistic_regression"]:.4f} ms for LR, '
        f'and {costs["mlp"]:.4f} ms for the MLP. {comparison} '
        'The batch size changes the practical cost comparison; a one-review refit cost should not be extrapolated by multiplying it by the number of arriving reviews.')
    warning_count = sum(bool(info['convergence_warnings']) for result in results
        for row in result['resources']['learning_updates']['rows'] for info in row['fit_diagnostics'])
    table = '\n'.join(rows)
    breakdown = '\n'.join(stages)
    skipped = results[0]['resources']['learning_updates']['skipped_batches']
    skipped_text = 'The small run omits batches without enough later memory reviews: ' + ', '.join(str(row['batch_size']) for row in skipped) + '.' if skipped else 'Every requested batch size was measured.'
    return rf'''### Next evaluation: per-sample incremental learning versus batch retraining

**Question.** Starting from the same reviewed history, how much compute and latency does each method need to incorporate newly reviewed incidents? This requested follow-up is now a completed experiment.

**Setup.** Each measurement starts with {initial} reviewed incidents ({max(config.budgets)} per class), using the same review IDs across methods. Batches contain {', '.join(str(batch) for batch in batches)} chronologically later memory-partition reviews. For each batch, every method encodes the new incidents, incorporates their labels, then predicts that batch with the updated memory or model. Final-test and validation reviews do not participate.

HDC normalizes each new hypervector and adds it to the labelled class accumulator; it does not revisit earlier incidents or optimize an encoder. LR and the MLP reuse the retained earlier feature vectors and perform **one full L-BFGS refit per arriving batch**, on the initial reviews plus that batch, with the already frozen regularization and model settings. Twenty new reviews therefore mean one refit on {initial + 20} reviews, rather than twenty separate refits; forty mean one refit on {initial + 40}.

Each repetition resets to the same starting state. Initial fitting and HDC state restoration occur outside the timer. All methods use one Torch/BLAS CPU thread, {config.refit_warmup} warm-up repetitions and {config.refit_repeats} measured repetitions per batch and seed pair, with a warm encoder cache. The measurements include feature/vector encoding, learning, and prediction; joins, geographic eligibility, audit persistence and database work remain separate. {skipped_text}

For $B$ new reviews and $N$ previously reviewed incidents, the measured operations are:

$$
T_{{\mathrm{{HDC}}}}(B)=T_{{\mathrm{{encode}}}}(B)+T_{{\mathrm{{normalize+add}}}}(B)+T_{{\mathrm{{predict}}}}(B),
$$

$$
T_{{\mathrm{{LR/MLP}}}}(N,B)=T_{{\mathrm{{encode}}}}(B)+T_{{\mathrm{{refit}}}}(N+B)+T_{{\mathrm{{predict}}}}(B),
\qquad t_{{\mathrm{{per\ new\ review}}}}=\frac{{T(N,B)}}{{B}}.
$$

The per-review number for LR/MLP is **amortized batch cost**. It is not the time to update immediately when each sample arrives; collecting a batch introduces a waiting time that this compute benchmark does not measure. For HDC, the same additions can be applied one review at a time without waiting for a batch. The delayed-feedback experiment demonstrates that separate behaviour.

**Result.** Total wall latency and amortized cost per new review, reported as the median of seed-pair medians. The p95 column is the median of seed-pair p95 measurements; it is descriptive timing variation, not an independent-world confidence interval. Process CPU time measures CPU work during the complete operation alongside elapsed wall time.

![Complete batch latency and amortized cost per newly reviewed incident](figures/learning-updates.png)

| New reviews | Learning method | Reviews after update | Total wall, ms | Wall p95, ms | Wall per new review, ms | CPU per new review, ms |
|---:|---|---:|---:|---:|---:|---:|
{table}

The corresponding stage costs per new review are:

| New reviews | Learning method | Encoding, ms | Addition or refit, ms | Prediction, ms |
|---:|---|---:|---:|---:|
{breakdown}

{finding}

Stage medians need not sum exactly to the median complete-operation latency. Raw repeated measurements and fit diagnostics are retained under `resources.learning_updates` in every pair's `results.json`. Timed HDC updates must exactly match a complete reconstruction from the initial and newly reviewed hypervectors; those checks occur outside the timer. There are {warning_count} convergence-warning refits among the measured batch repetitions; diagnostics are retained rather than silently discarded.

**Interpretation.** This compares additive HDC class-memory updates with these implementations' full batch retraining strategy, including the cost of encoding new incidents. It does not benchmark incremental SGD, warm starts, cached encoder-free updates, energy or a production pipeline. HDC's update work depends on the incoming hypervectors and fixed class memory, while a batch refit consumes the growing retained review set. Because the initial history is fixed at {initial} reviews, the experiment measures the requested batch costs; it does not establish a scaling law over arbitrarily large training histories.
'''


def encoder_explanation(config):
    explanation = r'''## How the encoder is built

The encoder turns an episode into one **@D@-dimensional hypervector**. Think of it as an additive description: a measurement contributes according to what it measures, where it sits in the dependency path, and when it occurs. The result retains those distinctions while supporting a single similarity comparison.

### 1. Resolve the facts before encoding

At each of three observations, exact temporal joins follow the phone's serving cell and that cell's valid backhaul edge. They supply six numeric facts. The table names the source of each fact so that “context” means a specific network relationship:

| Source at the observation time | Measurement | Physical range used for encoding |
|---|---|---|
| The commuter's phone | Received cellular signal strength | −125 to −65 dBm |
| The cell serving that phone | Cell load | 0–100% |
| The backhaul link used by that cell | Packet loss | 0–10% |
| The same backhaul link | Latency | 0–120 ms |
| Peer phones sharing that backhaul link | Mean received signal strength | −125 to −65 dBm |
| The same peer phones | Mean packet loss | 0–10% |

The **local channel** contains the commuter phone's own signal measurement. The **connected-context channel** contains the other five measurements: cell load, backhaul loss and latency, and the two peer averages, all joined through the active dependency at the same time. Three observation times produce three local facts and fifteen context facts per episode. The phone's handset model is a separate, small categorical contribution.

The graph chooses which source measurements enter the representation. The encoder binds **typed roles**, such as phone → serving cell → backhaul; it does not bind subscriber, cell or link identifiers. Two unrelated subscribers can therefore resemble each other when their measurements and connected context match. Changing an edge can change the joined facts even when the full network contains the same measurements.

### 2. Make nearby numbers similar

For a measurement $x$ with physical range $[a_f,b_f]$, first map it to a clipped fraction, then to one of @K@ numeric levels:

$$
s_f(x)=\operatorname{clip}\!\left(\frac{x-a_f}{b_f-a_f},0,1\right),
\qquad q_f(x)=\operatorname{round}\!\left((K-1)s_f(x)\right).
$$

TorchHD supplies a seeded family of correlated level hypervectors $\ell_0,\ldots,\ell_{K-1}$. Adjacent levels overlap more than distant levels. For example, −90 and −92 dBm receive nearby representations, while −90 and −115 receive more distinct ones. The ranges are fixed physical bounds, not statistics fitted to validation or test data. Values outside them are clipped; rounding introduces finite numeric resolution.

The same level family serves every numeric field. Binding each value to its attribute role distinguishes radio strength from packet loss, even if their scaled fractions coincide.

### 3. Bind the value to its meaning and connected role

We use $\otimes$ for **binding**, $\oplus$ for **bundling**, and $\rho$ for **permutation**. Here binding multiplies corresponding coordinates, bundling adds corresponding coordinates without thresholding, and $\rho^j(v)$ rotates the coordinates of $v$ by $j$ positions. The repeated-bundling symbol $\bigoplus$ combines several hypervectors. Atomic roles and channel markers are seeded bipolar arrays containing +1 and −1.

For local radio, the role product is:

$$
P_{\mathrm{radio}}=r_{\mathrm{phone}}\otimes r_{\mathrm{radio}}.
$$

For packet loss on the connected link, it includes the ordered dependency roles:

$$
P_{\mathrm{loss}}=
r_{\mathrm{phone}}
\otimes\rho^1\!\left(r_{\mathrm{serves}}\right)
\otimes\rho^2\!\left(r_{\mathrm{cell}}\right)
\otimes\rho^3\!\left(r_{\mathrm{uses}}\right)
\otimes\rho^4\!\left(r_{\mathrm{backhaul}}\right)
\otimes\rho^4\!\left(r_{\mathrm{loss}}\right).
$$

The rotations mark positions along the typed path. Peer facts extend it with shared-dependency and peer-phone roles. This role product specifies the meaning of the joined measurement; source identities remain in the accompanying manifest for provenance.

### 4. Preserve observation order, then bundle the channels

For observation position $p\in\{0,1,2\}$ and field $f$, the unweighted contribution is:

$$
e_{p,f}=\rho^{p+1}\!\left(c_{\mathrm{channel}(f)}\otimes P_f\otimes\ell_{q_f(x_{p,f})}\right).
$$

The outer rotation marks **before, during or after**. It is separate from the rotations marking path roles. Moving a radio measurement from before the disruption to after it changes its contribution, allowing deterioration and recovery to remain distinguishable.

Bundling adds these contributions. Define the local and context channels as:

$$
L=\frac{1}{\sqrt{3}}\bigoplus_{p=0}^{2}e_{p,\mathrm{radio}},
\qquad
C=\frac{1}{\sqrt{15}}\bigoplus_{p=0}^{2}\bigoplus_{f\in\mathcal F_C}e_{p,f}.
$$

The square-root divisors account for the different numbers of terms; they do not force every episode's channel norm to be equal. If $H$ is the handset-category hypervector bound to its attribute and channel roles, the raw episode accumulator is:

$$
z=(w_L L)\oplus(w_C C)\oplus(w_H H),
\qquad (w_L,w_C,w_H)=(@LOCAL@,@CONTEXT@,@HANDSET@).
$$

**“Context weight” means $w_C$, the multiplier of the bundled connected-context channel $C$ before normalization.** Context here consists of the five graph-joined network and peer measurements in the table, at each of three times; it does not mean location, subscriber identity or free text. With the active defaults, the complete equation is:

$$
z=\frac{1}{\sqrt{3}}\bigoplus_{p=0}^{2}e_{p,\mathrm{radio}}
 \oplus\frac{2}{\sqrt{15}}\bigoplus_{p=0}^{2}\bigoplus_{f\in\mathcal F_C}e_{p,f}
 \oplus0.25H.
$$

Each radio fact therefore has raw coefficient $1/\sqrt{3}\approx0.577$, while each network-context fact has $2/\sqrt{15}\approx0.516$. The channel multiplier is two, but each context fact does not receive twice the coefficient of a radio fact: the channel contains more terms. The experiment configuration and stored term manifests record these coefficients explicitly.

**Why give connected context more weight?** A shared transport incident can accompany weak, recovering or normal phone radio. Its common evidence lies upstream. Weight @CONTEXT@ was chosen in the earlier validation diagnosis and frozen before this fresh evaluation. It keeps the network evidence from being overwhelmed by the varying radio profile. It is a modelling choice for this study, not a universal HDC constant.

These are weights on raw contributions. Doubling a channel multiplies its direct contribution to a pairwise dot product by four before normalization; cross terms and normalization also affect the final score. It does not reserve a fixed percentage of similarity for that channel.

### 5. Use the same accumulator for retrieval, editing and learning

All arithmetic above uses float32 and retains the unthresholded bundle. Only then normalize:

$$
\hat z=\frac{z}{\lVert z\rVert_2},
\qquad \operatorname{similarity}(q,x)=\hat z_q^\mathsf{T}\hat z_x.
$$

Retrieval first applies exact spatial/time eligibility, then ranks eligible earlier episodes by similarity. Search storage uses float16; computation returns to float32 and the stored-vector residual is checked separately.

Because the raw bundle is retained, removing the handset means $z'=z\oplus(-w_HH)$, followed by normalization. Subtracting a component from an already normalized vector would be a different operation. The acceptance checks compare this edit with a complete rebuild.

A reviewed incident labelled $y$ updates a class accumulator by addition:

$$
A_y\leftarrow A_y\oplus\hat z,
\qquad \operatorname{score}_y(q)=\hat z_q^\mathsf{T}\frac{A_y}{\lVert A_y\rVert_2}.
$$

This is an **additive cosine class-memory classifier**: one accumulator per class, containing the sum of normalized hypervectors from that class's reviewed incidents. Prediction chooses the available class whose normalized accumulator has the highest cosine similarity to the query. The class representative is sometimes called a prototype; it is specifically this accumulated vector, not a separate feature model or neural network. The encoder stays fixed while labelled memory grows. Unseen classes are excluded from prediction; before any reviews, the system reports insufficient labelled memory. Updates retain an audit record and support exact reversal.

For an inspected candidate with weighted terms $t_j$, the retained manifest also permits exact arithmetic attribution:

$$
z_x=\bigoplus_{j=1}^{m} t_j,
\qquad a_j=\frac{\hat z_q^\mathsf{T}t_j}{\lVert z_x\rVert_2},
\qquad \operatorname{similarity}(q,x)=a_1+\cdots+a_m.
$$

The $t_j$ are hypervector contributions, combined by bundling; each $a_j$ is a scalar contribution to the cosine score. Each contribution links back to source observations, telemetry and valid edges. These contributions explain how the numeric score was assembled, including interference between terms. They do not establish the cause of a dropped call: the source records provide provenance, while similarity proposes comparisons.
'''
    for marker, value in {'@D@': f'{config.dimension:,}', '@K@': config.levels,
                          '@LOCAL@': config.local_weight, '@CONTEXT@': config.context_weight,
                          '@HANDSET@': config.handset_weight}.items():
        explanation = explanation.replace(marker, str(value))
    return explanation


def create_report(run_dir):
    from study import code_hash, verify_prepared
    run_dir = Path(run_dir)
    config = Config.read(run_dir / 'config.json')
    manifest = read(run_dir / 'manifest.json')
    verify_prepared(run_dir, config)
    if manifest['stage'] != 'complete' or len(manifest['completed_pairs']) != len(config.data_seeds) * len(config.encoder_seeds):
        raise ValueError('A complete experiment grid is required before reporting')
    if manifest['run_code_hash'] != code_hash() or manifest['summary_hash'] != file_hash(run_dir / 'summary.json'):
        raise ValueError('Code or summary changed after the recorded experiment')
    for name, checksum in manifest['artifact_hashes'].items():
        if file_hash(run_dir / name) != checksum:
            raise ValueError(f'Report artifact changed: {name}')
    paths = [run_dir / 'pairs' / f'data-{data}_encoder-{encoder}' for data in config.data_seeds for encoder in config.encoder_seeds]
    for path, (data, encoder) in zip(paths, [(d, e) for d in config.data_seeds for e in config.encoder_seeds]):
        if file_hash(path / 'results.json') != manifest['result_hashes'][f'{data}/{encoder}']:
            raise ValueError('Pair result checksum mismatch')
    results = [read(path / 'results.json') for path in paths]
    summary = read(run_dir / 'summary.json')
    case = read(paths[0] / 'case.json')
    online = read(paths[0] / 'online.json')
    geography = read(run_dir / 'data' / str(config.data_seeds[0]) / 'manifest.json')['geography']
    root = run_dir / 'reports'
    source_link = os.path.relpath(ROOT / 'src', root)
    make_figures(root / 'figures', summary, results, case, config, run_dir / 'geography')
    retrieval_rows = '\n'.join(f'| {NAMES[method]} | {interval_text(summary["retrieval"][method]["precision_at_5"])} | {percent(summary["retrieval"][method]["top1"]["mean"])} | {summary["retrieval"][method]["reciprocal_rank_at_10"]["mean"]:.3f} |' for method in ('hdc', 'structured', 'hdc_without_paths', 'hdc_without_order', 'hdc_lance_float16'))
    learning_rows = '\n'.join('| ' + str(budget) + ' | ' + ' | '.join(interval_text(summary['learning'][str(budget)][method]) for method in ('hdc', 'logistic_regression', 'mlp')) + ' |' for budget in config.budgets)
    error_budget = 5 if 5 in config.budgets else min(config.budgets)
    class_error_rows = '\n'.join('| ' + PATTERNS[label].capitalize() + ' | ' + ' | '.join(
        percent(1 - statistics.mean(row['metrics']['per_class'][label]['recall']
            for result in results for row in result['learning']
            if row['budget_per_class'] == error_budget and row['method'] == method))
        for method in ('hdc', 'logistic_regression', 'mlp')) + ' |' for label in LABELS)
    fit_rows = '\n'.join('| ' + str(budget) + ' | ' + ' | '.join(
        f'{statistics.median(next(row for row in result["learning"] if row["budget_per_class"] == budget and row["method"] == method)["fit"]["fit_ms"] for result in results):.2f}'
        for method in ('hdc', 'logistic_regression', 'mlp')) + ' |' for budget in config.budgets)
    classifier_bytes = {method: statistics.median(next(row for row in result['learning']
        if row['budget_per_class'] == max(config.budgets) and row['method'] == method)['fit']['persistence']['bytes']
        for result in results) for method in ('logistic_regression', 'mlp')}
    warning_count = sum(bool(row['fit']['convergence_warnings']) for result in results for row in result['learning'] if row['method'] != 'hdc')
    frozen = read(run_dir / 'frozen_settings.json')
    trial_fits = [info for trial in frozen['trials'] for info in trial['fit_diagnostics']]
    selected_rows = '\n'.join(f'| {budget} | {frozen["parameters"][str(budget)]["logistic_regression"]["C"]} | {frozen["parameters"][str(budget)]["logistic_regression"]["scaling"]} | {frozen["parameters"][str(budget)]["mlp"]["width"]} | {frozen["parameters"][str(budget)]["mlp"]["alpha"]} | {frozen["parameters"][str(budget)]["mlp"]["scaling"]} |' for budget in config.budgets)
    trial_warning_count = sum(bool(info['convergence_warnings']) for info in trial_fits)
    learning_difference = summary['learning_differences'][str(error_budget)]
    difference_text = '; '.join(f'HDC minus {name}: {learning_difference["hdc_minus_" + method]["mean"] * 100:.1f} percentage points [{learning_difference["hdc_minus_" + method]["lower"] * 100:.1f}, {learning_difference["hdc_minus_" + method]["upper"] * 100:.1f}]'
        for method, name in [('logistic_regression', 'LR'), ('mlp', 'MLP')])
    timing_rows = '\n'.join(f'| {name} | {statistics.median(result["resources"]["timings"][key]["median_ms"] for result in results):.4f} ms | {statistics.median(result["resources"]["timings"][key]["p95_ms"] for result in results):.4f} ms |' for key, name in [('hdc_add_only', 'HDC addition only'), ('hdc_encode', 'HDC encoding'), ('hdc_score', 'HDC scoring'), ('hdc_encode_predict', 'HDC encode–predict'), ('logistic_regression_encode_predict', 'LR encode–predict'), ('mlp_encode_predict', 'MLP encode–predict'), ('hdc_encode_score_update', 'HDC encode–score–update'), ('logistic_regression_review_refit', 'LR encode–refit–predict'), ('mlp_review_refit', 'MLP encode–refit–predict')])
    counts = results[0]['candidate_counts']
    path_gain = summary['paired_differences']['hdc_minus_hdc_without_paths']
    order_gain = summary['paired_differences']['hdc_minus_hdc_without_order']
    best_budget = max(config.budgets)
    last_f1 = summary['learning'][str(best_budget)]['hdc']['mean']
    learning_direction = 'More reviews improve the aggregate result over the one-example starting point.' if last_f1 > summary['learning'][str(min(config.budgets))]['hdc']['mean'] else 'More reviews do not improve the aggregate result over the one-example starting point.'
    budget_80 = next((budget for budget in config.budgets if summary['learning'][str(budget)]['hdc']['mean'] >= .8), None)
    low_sample = f'HDC reaches mean macro F1 ≥0.80 with {budget_80} ' + ('review' if budget_80 == 1 else 'reviews') + f' per class ({budget_80 * 4} total).' if budget_80 else f'HDC does not reach mean macro F1 0.80 within {best_budget} reviews per class.'
    effect_examples = {}
    for name, predicate in [('helps', lambda row: row['net_correct'] > 0), ('hurts', lambda row: row['net_correct'] < 0), ('unchanged', lambda row: row['changed'] == 0)]:
        found = next((row for row in online['effects'] if predicate(row)), None)
        effect_examples[name] = found
    save_json(root / 'update_examples.json', effect_examples)
    effect_text = '\n'.join(f'- **{name.capitalize()}:** ' + (f'{row["episode_id"]}: {row["corrected"]} corrected, {row["regressed"]} regressed, {row["changed"]} ' + ('prediction changed.' if row['changed'] == 1 else 'predictions changed.') if row else 'No example occurred in the first fixed replay; none was manufactured.') for name, row in effect_examples.items())
    mismatches = len(read(paths[0] / 'retrieval_errors.json'))
    errors = read(paths[0] / 'retrieval_errors.json')[:5]
    error_text = '\n'.join(f'| {row["episode_id"]} | {PATTERNS[row["label"]]} | {PATTERNS[row["retrieved_label"]]} |' for row in errors) or '| — | No top-1 mismatch in this run | — |'
    operator = results[0]['operators']
    claim_rows = [
        ('Composable representation', f'Role swaps, order changes and edge rewiring are detectable; controls without the relevant operator remain invariant.', 'Controlled illustrations and ablations support the designed representation, not arbitrary graph reasoning.'),
        ('Inspectable evidence', f'All {sum(r["evidence_audit"]["top_hits_checked"] for r in results):,} retrieved top hits resolve to source records and reconstruct their float32 reference scores.', 'Provenance is retained alongside vectors; arithmetic contributions include interference and do not establish causes. Float16 search has a separately recorded quantization residual.'),
        ('Online learning', f'Frozen encoder, delayed reviews, additive class memories, exact reversal; {low_sample}', 'Supervised learning still requires labels. Updates can regress, and the scenarios are simulated.'),
        ('Compute cost', 'Addition, prediction and review incorporation are measured separately on CPU against trained LR and MLP.', 'Fast addition alone is not serving latency, energy efficiency, or an advantage over the measured controls.'),
        ('Storage', f'{results[0]["resources"]["hdc_raw_vector_bytes"]:,} bytes per raw hypervector versus {results[0]["resources"]["explicit_feature_bytes"]} bytes per explicit feature vector.', 'This study shows no storage saving from HDC.')]
    claims = '\n'.join(f'| {name} | {evidence} | {boundary} |' for name, evidence, boundary in claim_rows)
    report = f'''# Hyperdimensional Computing for Telecom Incident Triage

*An experimental study of composable representations, retrieval with provenance, and incremental learning*

## The operational problem

Telecom service assurance concerns the quality of the calls and data sessions customers use. When service degrades, an operations team needs to decide where to investigate first and how widely the problem may extend. A symptom affecting one subscriber might involve the phone's radio connection; similar symptoms across several subscribers might involve a shared network dependency. This first assessment is **incident triage**. It guides investigation before a cause is confirmed.

The evidence spans several kinds of data: subscriber and handset records, measurements over time, network telemetry, locations, and the relationships between network components. A useful comparison must connect those records. The same measurement can mean different things depending on which component produced it, which other components depend on it, and whether service is worsening or recovering.

Earlier incidents offer a practical starting point. A triage tool can bring comparable cases to an engineer's attention, show the observations and network facts supporting the comparison, and incorporate the outcomes of reviewed cases into future decisions. This creates a combined data-engineering and learning problem: preserve the meaning of connected, changing evidence while maintaining a memory that can grow as reviews arrive.

## Scope and research questions

We narrow that broader use case to one question:

> A commuter's call drops. Which earlier incidents resemble it, what network facts support that comparison, and how can reviewed incidents improve future triage?

The study evaluates two tasks: retrieving comparable earlier incidents, and assigning a triage pattern using previously reviewed examples. Each incident includes a short sequence of phone measurements and the network context connected to that phone at those times. Triage happens after the full observation window, so the available evidence can include deterioration or recovery.

**Hyperdimensional computing (HDC)** represents these facts in long numeric arrays called hypervectors. Its operations can attach a value to a role, combine contributions, and preserve order. We study whether an episode built with those operations can support retrieval, inspection of its source evidence, and learning through additions to labelled class memory while the encoder stays fixed.

The experiments examine four questions:

- **Representation:** Do roles, event order and network connectivity change the meaning of an encoded incident?
- **Retrieval and evidence:** Can it find comparable earlier incidents and return the source records supporting the comparison?
- **Learning:** How useful does class memory become as reviewed examples arrive, and when do updates help or hurt?
- **Resources:** What do encoding, prediction, learning updates, retrieval and storage cost, including the cost per new review?

An explicit structured comparator tests retrieval using the same ordered measurements and connected context. Regularized logistic regression and a small MLP provide familiar trained-model comparisons for triage-pattern prediction. All three learners receive the same reviewed incidents. These comparisons determine the practical tradeoffs; the operator demonstrations explain how the representation works.

## The dataset

We use a controlled simulation with a recognizable geographic setting: a commute corridor along Bloor Street West in Toronto. Public street geometry supplies the map. Every subscriber, handset, trajectory, network component, measurement and review is simulated. This lets us vary roles, chronology and connectivity deliberately, and check results against retained source records. It does not establish performance on a carrier's operational data.

### Just enough telecom to read the example

A **subscriber** is the simulated customer; a **handset** is their phone. **Radio** means the wireless part of its connection. The **serving cell** provides that phone's cellular connection at the observation time, through base-station equipment. A **backhaul link** carries traffic from the cell into the rest of the operator's network. The simplified path is **phone → serving cell → backhaul link**. Other phones can use different cells but share that link; we call them **peer phones**. This [radio/transport distinction](https://www.ericsson.com/en/public-policy-and-government-affairs/5-key-facts-about-5g-radio-access-networks) is what makes connected evidence useful.

| Measurement | What it tells us |
|---|---|
| Received signal strength, in dBm | How much cellular signal power reaches the phone. **−80 dBm is stronger than −90 dBm.** |
| Cell load, in % | How busy the serving cell is; a simulated utilization indicator in this study. |
| Packet loss, in % | The share of data packets that fail to arrive. A packet is a small chunk of transmitted data; 1% loss means about one in 100 is missing. |
| Latency, in ms | How long data takes to travel over the link. One millisecond is 0.001 seconds. |
| Peer averages | Mean signal-strength and loss measurements from phones sharing the same backhaul link at the same time. |

**dBm** expresses power on a logarithmic scale relative to one milliwatt. Negative values mean power below that reference, not negative power. A 10 dB decrease means ten times less power. We simulate a generic received-power measurement; signal strength alone does not determine call quality. [Unit reference](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1sl378/1SL378_0e_ColdSrcNF.pdf).

### The connected records

![Conceptual graph: a subscriber's phone, ordered observations, and two cells sharing a backhaul link, with node and edge properties](figures/schema.png)

[Open the schema at full size](figures/schema.svg) or [in a browser](figures/schema.html). Circles are nodes; directed arrows are relationships reconstructed from record IDs. The property values are illustrative. The phone → cell relationship comes from its observation at time $t$; cell → link edges are valid when $t_0 ≤ t < t_1$. Cell load and link measurements marked **(t)** are joined telemetry at that time, rather than static component properties. Peer observations are separate source records; they need not identify individual peer subscribers.

Each simulated network/time block contains six cells and two backhaul links with one consistent **telemetry** timeline: measurements describing their state over time. Phones using the same component at the same time see the same infrastructure state. The cell-to-link edges change halfway through the block, with recorded validity intervals. Subscriber identities connect records; they do not determine similarity.

The unit of analysis is an **episode**: three observations at 0, 20 and 40 seconds, before, during and after a disruption or a normal-service reference window. Four balanced patterns define the evaluation classes:

| Pattern | What the observations show |
|---|---|
| Deteriorating radio | The phone's received signal becomes weaker across the episode. |
| Transient recovery | The phone's received signal improves by the end of the episode. |
| Shared transport impairment | The shared backhaul link and peer phones show packet loss; the phone's own signal profile can vary. |
| Normal service | Neither a large signal change nor the shared backhaul impairment occurs. |

An independent checker reconstructs these labels from source measurements, event order and valid edges. Encoder input does not include generator scenario names, labels or service-disruption outcomes. Review labels become available {config.review_delay_s} seconds after the complete episode; they stand in for analyst feedback in the demonstration. No human reviews were collected.

Each data seed produces {config.blocks * config.episodes_per_block:,} episodes across {config.blocks} independent network/time blocks. Of those blocks, {config.memory_blocks} build labelled memory, {config.validation_blocks} support model selection, and {config.blocks - config.memory_blocks - config.validation_blocks} are reserved for final testing. Subscribers and shared incidents stay within their assigned block. No validation or final-test label updates memory.

The full experiment crosses {len(config.data_seeds)} data seeds with {len(config.encoder_seeds)} encoder seeds. It scores {summary['final_queries_scored']:,} final queries across repetitions, representing {summary['independent_final_worlds']} distinct final-test worlds. Repeated encoders reuse episodes and are averaged before uncertainty intervals resample data seeds and complete worlds. Those intervals describe this generator.

The street snapshot contains {geography['corridor_segments']} Bloor Street segments, with frozen source checksums and longitude/latitude coordinates (EPSG:4326). No distances are computed from angular coordinates, and the simulation does not predict signal strength from the street geometry. {geography['attribution']} [Dataset]({geography['dataset_url']}); [licence]({geography['licence_url']}).

## Start with one incident

![A simulated commute, three signal-strength observations, and the connected network facts](figures/incident.png)

Consider a simulated commuter travelling along Bloor Street West. In this walkthrough, the phone's received signal strength changes from {case['query']['observations'][0]['radio_dbm']:.1f} dBm before the disruption to {case['query']['observations'][2]['radio_dbm']:.1f} dBm afterwards. More negative dBm values indicate weaker received signal strength. At the middle observation, the connected backhaul link reports {case['query']['observations'][1]['link_loss_pct']:.1f}% packet loss, and peer phones using that dependency report {case['query']['observations'][1]['peer_loss_pct']:.1f}% mean loss.

These facts make the comparison specific: look for an earlier episode with a similar signal-strength trajectory and connected network conditions. Exact spatial and time filters first identify eligible earlier memory episodes. Similarity then ranks them, and retained observations, telemetry and valid edges let the engineer inspect what the retrieved episode has in common.

The query is **{case['query']['episode_id']}**, whose independently reconstructed pattern is **{PATTERNS[case['query_label']]}**. Its first retrieved comparison is **{case['candidate']['episode_id']}**, labelled **{PATTERNS[case['candidate_label']]}**. The walkthrough uses the first final-test episode with an observed service disruption, selected before inspecting retrieval correctness. It illustrates the workflow; the experiments below evaluate all final-test queries.

A retrieved comparison is evidence for further inspection. The triage labels describe the observed patterns in this simulation; they do not confirm the cause of the commuter's dropped call. The next section shows how the phone and network facts become one composable representation, before we measure retrieval and learning performance.

{encoder_explanation(config)}

## Experiment 1: what do the operators preserve?

**Question.** Can the representation distinguish the same values attached to different meanings, appearing in different orders, or connected through different edges?

**Setup.** Binding attaches a value to a role. Bundling adds contributions. Permutation marks an observation's place in the episode. The local channel has weight {config.local_weight:g}; connected cell/link/peer context has weight {config.context_weight:g}. Numeric levels preserve neighbourhoods; the handset contribution has weight {config.handset_weight:g}. The dimension is {config.dimension:,}.

Hyperdimensional computing (HDC) represents information in long numeric arrays called hypervectors. Here, each atomic role starts as a seeded array of +1 and −1 values. Distinct roles have little overlap, while nearby numeric measurements deliberately receive correlated arrays. An episode becomes a sum of these encoded contributions rather than an opaque identifier.

In this implementation, binding multiplies arrays element by element, bundling adds them, and permutation rotates their coordinates by a fixed number of positions. Binding gives radio strength a different meaning from link loss. A position-specific rotation distinguishes the same observations in reverse order. The connected channel follows the time-valid phone → cell → backhaul dependency before encoding its measurements and peer context.

The operator vocabulary is small: `bind(role, value)` keeps a measurement attached to its meaning; `bundle(facts)` combines contributions; `permute(observation, position)` marks order. Learning reuses addition: `class_memory += normalize(episode_vector)`. The encoder implements these operations with TorchHD; no text embedding model is needed for these structured records.

**Result.** Swapping good/bad states between phone radio and an upstream link gives cosine {operator['role_binding']['bound_cosine']:.3f}. Omitting binding makes the two accumulators equal within {operator['role_binding']['without_binding_max_error']:.2g}. Reversing the observations gives cosine {operator['event_order']['with_order_cosine']:.3f}; omitting order leaves error {operator['event_order']['without_order_max_error']:.2g}. Rewiring one serving edge gives cosine {operator['connectivity']['with_path_cosine']:.3f}; pooling the same network measurements without their connections leaves error {operator['connectivity']['without_path_max_error']:.2g}. The adjacent numeric-level cosine is {operator['numeric_neighbourhood']['adjacent_cosine']:.3f}, compared with {operator['numeric_neighbourhood']['far_cosine']:.3f} for distant levels.

**Interpretation.** These controlled illustrations show exactly what the operators do. The graph pair retains every measured value and changes one edge in a separate counterfactual world. Its two-candidate ranking has a 50% random reference and serves as an illustration, not a practical graph benchmark. Broader retrieval usefulness is measured next.

The handset contribution can be removed from a raw query without re-encoding the other facts. Reconstruction error in the walkthrough is {case['query_edit_max_error']:.2g}. The original top five are `{', '.join(case['original_top5'])}`; after removing handset they are `{', '.join(case['without_handset_top5'])}`. A changed ranking is an editable query, not automatically a better result.

## Experiment 2: can it retrieve useful earlier incidents?

**Question.** Does the representation return earlier incidents with the same independently checked triage pattern, and can their source evidence be inspected?

**Setup.** Every query uses an exact corridor/time filter and a memory-only candidate set. All three observations must qualify. Lance and independently registered GeoDataFusion agree on selected observation identities. The first run has {counts['min']}–{counts['max']} candidates per query (mean {counts['mean']:.1f}). Relevance means the same triage pattern, not the same real-world cause. The numeric comparator receives the same ordered measurements and connected context as HDC. Connectivity and order are removed separately in ablations.

![Retrieval precision with grouped uncertainty intervals](figures/retrieval.png)

| Method | Precision@5, 95% interval | Top-1 match | Reciprocal rank@10 |
|---|---:|---:|---:|
{retrieval_rows}

Random ranking yields expected precision {percent(summary['random_precision_at_5'])}, based on each query's eligible label prevalence. The paired HDC difference when connectivity is omitted is {path_gain['mean'] * 100:.1f} percentage points [{path_gain['lower'] * 100:.1f}, {path_gain['upper'] * 100:.1f}]. When order is omitted, the difference is {order_gain['mean'] * 100:.1f} points [{order_gain['lower'] * 100:.1f}, {order_gain['upper'] * 100:.1f}].

**Evidence.** Every stored-vector top hit was checked against its source records. Maximum raw reconstruction error across runs is {max(r['evidence_audit']['max_raw_reconstruction_error'] for r in results):.2g}; maximum float32 reference-score reconstruction error is {max(r['evidence_audit']['max_score_reconstruction_error'] for r in results):.2g}. Reference scores are decomposed into additive term contributions under the raw accumulator's normalization. A separately recorded quantization residual connects that reference to the stored float16 vector's cosine. These contributions include cross-term interference; they are not causal importance scores. Source identities and timestamps support each term.

**Errors.** The first fixed run has {mismatches} top-1 mismatches out of {results[0]['n_queries']}. Examples are retained rather than discarded:

| Query | Expected pattern | Retrieved pattern |
|---|---|---|
{error_text}

**Interpretation.** Compare HDC with the explicit comparator directly. The ablations reveal the value of the information represented by order and connected context, not an exclusive HDC capability. The comparator can represent those facts too. Float16 search storage is independently verified against its quantized vectors; its mean precision difference from float32 is {statistics.mean(r['float16_quantization']['precision_at_5_change'] for r in results) * 100:.3f} points. Float32 accumulators remain authoritative for arithmetic and updates.

## Experiment 3: how does memory grow through reviews?

**Question.** How useful is class memory with few labelled incidents, and what happens when feedback arrives after a decision?

**Setup.** Each class starts with no vector. Encoding produces an episode hypervector; a review adds its normalized vector to the appropriate float32 class accumulator. Prediction compares the episode with normalized class memories. The comparison models are regularized multinomial logistic regression and a small one-hidden-layer ReLU MLP. Both receive the same 21 explicit features: six measurements at each of three ordered observations, plus three handset-category indicators. The connected measurements come from the same valid graph joins as HDC. Validation chooses between the already declared physical range scaling and an additional StandardScaler fitted only to the reviewed memory examples at each budget. Both models use L-BFGS optimization to a declared tolerance; they are not limited to one training pass. LR regularization and MLP width/regularization are selected separately at each budget using the first data seed's validation worlds, averaged across three initialization seeds, then frozen for every final test. Every method receives the same reviewed incidents. Review budgets count training labels; additional labelled validation worlds support model selection.

![Learning from reviewed incidents with no encoder retraining](figures/learning.png)

| Reviews per class | HDC macro F1 | Trained LR macro F1 | Small MLP macro F1 |
|---:|---:|---:|---:|
{learning_rows}

**Result.** {learning_direction} {low_sample} The curve can plateau or regress at intermediate budgets; more reviews do not guarantee improvement. Macro F1 gives each pattern equal weight, with 1.0 representing perfect classification. The first delayed-feedback replay's HDC accuracy is {percent(online['metrics']['hdc']['accuracy'])}, with prediction coverage {percent(online['metrics']['hdc']['coverage'])}. Coverage includes initial decisions for which no review has arrived; these produce “insufficient labelled memory”. This replay uses memory-building episodes, while the learning curves above use independent final-test worlds.

At {error_budget} reviews per class, the paired, world-grouped differences are {difference_text}. There are {warning_count} convergence warnings among the {len(results) * len(config.budgets) * 2} final LR/MLP fits, and {trial_warning_count} among {len(trial_fits)} validation-selection fits. Diagnostics and iteration counts are retained for every fit.

The aggregate score can hide a difficult pattern. At {error_budget} reviews per class, the following share of final-test examples is assigned to the wrong class, averaged across the seed grid. These are descriptive per-class errors; the grouped uncertainty intervals above apply to macro F1. Detailed confusion matrices are retained for every budget and world.

| Actual pattern | HDC error rate | Trained LR error rate | Small MLP error rate |
|---|---:|---:|---:|
{class_error_rows}

Updates are not guaranteed to help. The first fixed replay contains these examples:

{effect_text}

These examples use a fixed diagnostic probe within the memory partition, including episodes that are later reviewed. They are neither held-out performance estimates nor a signal for choosing updates. All updates follow the configured review schedule. The full log records review provenance, availability time, encoder hash, update-vector hash and before/after accumulator hashes. Exact reversal restores the previous float32 snapshot rather than relying on rounded subtraction.

**Interpretation.** There is supervised learning, but no encoder retraining or optimization loop for the HDC memory. Adding unlabelled records to a retrieval store is a different operation. The learning curves determine how strongly we can describe sample efficiency on these patterns; they do not establish it across telecom tasks.

## Experiment 4: how cheap is the complete operation?

**Question.** What does it cost to encode an episode, score it and update memory, and how much storage is used?

**Setup.** One Torch/BLAS CPU thread; batch one; {config.warmup} warm-up iterations and {config.benchmark_repeats} measurements for prediction/addition per seed pair. More expensive classifier refits use {config.refit_warmup} warm-ups and {config.refit_repeats} measurements. The table reports the median of run medians and the median of run p95 measurements. The runtime is {manifest['runtime']['platform']}, Python {manifest['runtime']['python']}. Joins, geo selection and database writes are separate from these warm compute measurements.

![Measured CPU operation costs, with addition distinguished from the full operation](figures/compute.png)

| Operation | Median of run medians | Median of run p95 |
|---|---:|---:|
{timing_rows}

HDC's raw accumulator uses {results[0]['resources']['hdc_raw_vector_bytes']:,} bytes per episode; its search vector uses {results[0]['resources']['hdc_search_vector_bytes']:,} bytes in float16. The explicit comparator uses {results[0]['resources']['explicit_feature_bytes']} bytes in float32. Four HDC class accumulators use {results[0]['resources']['class_memory_bytes']['hdc']:,} bytes, before mappings, counters, audit history and exact-undo snapshots. These sizes describe representations, not the complete trained models. This dataset supports no compression claim.

Measured resource totals in the first fixed run:

| Resource | Size |
|---|---:|
| All float32 HDC episode tensors | {results[0]['resources']['hdc_representation_tensor_bytes'] / 2**20:.2f} MiB |
| All explicit feature tensors | {results[0]['resources']['explicit_representation_tensor_bytes'] / 2**20:.2f} MiB |
| Cached encoder basis tensors | {results[0]['resources']['encoder_cached_basis_bytes'] / 2**20:.2f} MiB |
| Lance vector store, raw/search vectors and manifests | {results[0]['resources']['vector_store_bytes'] / 2**20:.2f} MiB |
| Lance source-record store | {results[0]['resources']['source_store_bytes'] / 2**20:.2f} MiB |

LR and MLP trained estimators, including any fitted scalers, are saved and checked for identical predictions after reloading. Their serialized sizes and initial fit times are recorded separately at every review budget.

At {max(config.budgets)} reviews per class, median serialized estimator size is {classifier_bytes['logistic_regression']:,.0f} bytes for LR and {classifier_bytes['mlp']:,.0f} bytes for MLP. These include estimator metadata and any fitted scaling, but exclude the retained review buffer. That buffer uses {results[0]['resources']['review_buffer_feature_and_label_bytes']:,} bytes for the refit measurement. HDC's class tensor size also excludes audit history and encoder basis tensors.

Observed median initial fit times across the seed grid, excluding encoding and persistence:

| Reviews per class | HDC memory building, ms | LR fitting, ms | MLP fitting, ms |
|---|---:|---:|---:|
{fit_rows}

These initial fits are recorded once per run and budget; the repeated refit benchmark above supplies a separate measurement of incorporating a further review. Model-selection compute is recorded in the frozen settings and is not included in these initial fits.

The directory totals include retained Lance versions and metadata. They describe these artifacts rather than an optimized storage comparison. Tensor totals exclude Python objects, source tables, temporary allocations and audit history; peak process memory was not measured.

The first run encodes all episodes in {results[0]['resources']['encode_all_s']:.3f} s, persists the vectors and manifests in {results[0]['resources']['persistence_s']:.3f} s, and performs all validation/test geography gates plus independent checks in {results[0]['resources']['exact_gate_s']:.3f} s. LanceDB retrieval median is {results[0]['resources']['retrieval_latency']['hdc_lance_float16']['median_ms']:.3f} ms; the matched in-memory HDC scoring-and-ranking median is {results[0]['resources']['retrieval_latency']['hdc']['median_ms']:.3f} ms. They measure different layers and are not an implementation speed contest.

**Interpretation.** The addition is cheap. For LR and MLP, review incorporation here means encoding the new incident, refitting on {results[0]['resources']['review_refit_training_examples']} retained reviewed examples, then predicting it. HDC encodes, predicts with its existing class memory, and adds the new review. Those are different update strategies. Conventional incremental optimizers and warm starts could reduce refit cost; they are not evaluated here, so this comparison supports no general claim that conventional ML must retrain from scratch. The complete operation and alternative methods deserve equal attention. These are component measurements on one machine, without energy, production serving, spatial-index or ANN benchmarks. Near-real-time suitability requires an application latency budget and its complete data path.

{learning_update_explanation(results, config)}

## What this supports

| Claim | Evidence | Boundary |
|---|---|---|
{claims}

The simulator uses simple observed rules and deliberately balanced classes, with complete telemetry. This makes the mechanisms observable and supplies a compact feature table that conventional trained classifiers can use effectively. It does not demonstrate operational diagnosis, realistic class prevalence, continual adaptation under drift, robustness to missing telemetry, or coverage across new incident types. Those would need separate experiments.

## Takeaways

In this controlled study, HDC combines role-sensitive, ordered representations with connected network evidence. Retrieval returns comparable earlier incidents with source records that can be inspected, and reviewed examples improve class memory through reversible additions while the encoder stays fixed.

HDC performs well with few reviewed examples. Its advantage is clearest at the smallest label budgets; LR and the MLP become competitive as more reviews arrive. The results support composable representation and incremental memory building, with usefulness measured against familiar trained classifiers.

The resource comparison depends on the update strategy and batch size. HDC incorporates individual reviews quickly, while LR batch retraining amortizes well and is cheaper per review at the largest measured batch. HDC uses more representation storage than the explicit feature vector. These findings apply to the simulated patterns and measured CPU workload; operational telemetry, missing data and drift remain untested.

## Reproduce and inspect

Inspect the [source code]({source_link}) for the implementation. Run `uv run src/prepare.py`, `uv run src/run.py`, then `uv run src/report.py` from the repository root. All three scripts use the paths in `src/settings.py`. The manifest records dependency versions, source checksums, configuration, encoder and code hashes, and all completed seed pairs. The committed metrics retain block-level results, per-class errors and repeated timing measurements. Reproducing a run generates the detailed predictions, review logs, retrieval errors, query edits and source witnesses locally. The active learning comparison uses trained LR and a small MLP.

The [LR](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) implementations come from locked scikit-learn 1.9.1. The encoder weighting was selected on separate validation worlds (data seeds 1001–1005) and frozen before this evaluation. The full study defaults now use fresh data seeds 1006–1010. These are new independent simulated worlds, not external carrier validation. The actual data seeds in this run are {', '.join(str(seed) for seed in config.data_seeds)}. LR/MLP settings use only this run's first data seed's validation worlds and are frozen before final testing.

The declared search space uses LR C ∈ {{0.1, 1, 10}}, MLP hidden width ∈ {{16, 32}}, MLP alpha ∈ {{0.001, 0.1}}, and fixed physical range scaling or an additional training-fitted StandardScaler. C controls inverse L2 regularization strength; alpha controls MLP regularization. The HDC weighting is fixed throughout this run; no HDC variant search, additional feature engineering or test-based selection is performed. Only the active encoder is evaluated, alongside operator ablations that answer the representation questions. The frozen choices are:

| Reviews/class | LR C | LR scaling | MLP width | MLP alpha | MLP scaling |
|---|---:|---|---:|---:|---|
{selected_rows}
'''
    (root / 'summary.md').write_text(report)
    save_json(root / 'claims.json', {'code_hash': manifest['run_code_hash'], 'report_generator_hash': file_hash(Path(__file__)),
        'summary_hash': manifest['summary_hash'],
        'figure_hashes': {str(path.relative_to(root)): file_hash(path)
                         for path in sorted((root / 'figures').iterdir()) if path.is_file()},
        'claims': [{'claim': a, 'evidence': b, 'boundary': c} for a, b, c in claim_rows],
        'report_hash': file_hash(root / 'summary.md')})
    print(f'Report written to {root / "summary.md"}', flush=True)
    return root / 'summary.md'
