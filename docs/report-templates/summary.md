# Hyperdimensional Computing for Telecom Incident Triage

*An experimental study of composable representations, retrieval with provenance, and incremental learning*

## The operational problem

Telecom service assurance concerns the quality of the calls and data sessions customers use. When service degrades, an operations team needs to decide where to investigate first and how widely the problem may extend. A symptom affecting one subscriber might involve the phone's radio connection; similar symptoms across several subscribers might involve a shared network dependency. This first assessment is **incident triage**. It guides investigation before a cause is confirmed.

The evidence spans several kinds of data: subscriber and handset records, measurements over time, network telemetry, locations, and the relationships between network components. A useful comparison must connect those records. The same measurement can mean different things depending on which component produced it, which other components depend on it, and whether service is worsening or recovering.

Earlier incidents offer a practical starting point. A triage tool can bring comparable cases to an engineer's attention, show the observations and network facts supporting the comparison, and incorporate the outcomes of reviewed cases into future decisions. This creates a combined data-engineering and learning problem: preserve the meaning of connected, changing evidence while maintaining a memory that can grow as reviews arrive.

## Scope and research questions

We narrow that broader use case to one question:

> A commuter's call drops. Which earlier incidents resemble it, what network facts support that comparison, and how can reviewed incidents improve future triage?

The study evaluates two tasks: retrieving comparable earlier incidents, and assigning a triage pattern using previously reviewed examples. Each incident includes a short sequence of phone measurements and the network context connected to that phone at those times. Triage happens after the full observation window, so the available evidence can include deterioration or recovery.

We study whether **hyperdimensional computing (HDC)** can represent these connected, time-ordered facts in a way that supports retrieval, inspection of the source evidence, and learning from reviewed incidents without retraining.

The experiments examine four questions:

- **Representation:** Do roles, event order and network connectivity change the meaning of an encoded incident?
- **Retrieval and evidence:** Can it find comparable earlier incidents and return the source records supporting the comparison?
- **Learning:** How many reviewed incidents does each method need to classify new incidents well?
- **Resources:** How long do prediction and learning from new reviews take, and how much storage does each representation need?

The learning and cost experiments compare HDC with regularized logistic regression (LR) and a small **multilayer perceptron (MLP)**, a neural network with one hidden layer that learns to classify incidents from their measurements. All three methods receive the same reviewed incidents and connected measurements. Retrieval evaluates HDC on its own, removing order and connectivity as internal checks.

## How HDC works here, in brief

HDC represents each incident as one long list of {{ dimension }} numbers, called a **hypervector**, built with simple arithmetic. Nothing in the encoder is learned from the reviewed incidents: its random hypervectors are fixed by a seed, and its few weights were chosen in advance.

1. **Every kind of fact gets its own random hypervector**, such as "phone signal", "backhaul link" or "packet loss". Random hypervectors this long are almost unrelated to one another, so each one works as a distinct label.
2. **Measurements get hypervectors too**, chosen so that nearby values, such as −90 and −92 dBm, receive similar ones.
3. **Binding** multiplies a value's hypervector by its role's. The result means "this value, in this role": a weak signal *at the phone* encodes differently from the same reading *in a peer phone's average*.
4. **Permutation** shifts a hypervector's coordinates to record when a fact was observed: before, during or after the disruption.
5. **Bundling** adds all of an incident's facts into one hypervector. The sum still resembles each of its parts, so incidents with similar facts in the same roles end up with similar hypervectors.

That single hypervector then does several jobs:

- **Retrieval:** comparable earlier incidents are found by comparing hypervectors with cosine similarity.
- **Explanation:** because the hypervector is a sum, each similarity score splits exactly into contributions from individual facts.
- **Editing:** a fact can be removed from a query by subtracting its contribution.
- **Learning:** each pattern's **class memory** is the sum of its reviewed incidents' hypervectors. A new incident gets the pattern whose class memory it most resembles, and learning from a new review is one more addition, with no retraining.

The [representation walkthrough](../../docs/representation-walkthrough.md) follows one incident through every step with real numbers; [How the encoder is built](#how-the-encoder-is-built) gives the equations.

## Key findings

**Where HDC adds value**

{{ key_strengths }}

**Tradeoffs and limits**

{{ key_tradeoffs }}

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
| Transient disruption and recovery | The phone's received signal improves by the end of the episode. |
| Shared transport impairment | The shared backhaul link and peer phones show packet loss; the phone's own signal profile can vary. |
| Normal service | Neither a large signal change nor the shared backhaul impairment occurs. |

An independent checker reconstructs these labels from source measurements, event order and valid edges. Encoder input does not include generator scenario names, labels or service-disruption outcomes. Review labels become available {{ review_delay_s }} seconds after the complete episode; they stand in for analyst feedback in the demonstration. No human reviews were collected.

Each data seed produces {{ episode_count }} episodes across {{ block_count }} independent network/time blocks. Of those blocks, {{ memory_blocks }} build labelled memory, {{ validation_blocks }} support model selection, and {{ test_blocks }} are reserved for final testing. Subscribers and shared incidents stay within their assigned block. No validation or final-test label updates memory.

The full experiment crosses {{ data_seed_count }} data seeds with {{ encoder_seed_count }} encoder seeds. It scores {{ final_query_count }} final queries across repetitions, representing {{ final_world_count }} distinct final-test worlds. Repeated encoders reuse episodes and are averaged before uncertainty intervals resample data seeds and complete worlds. Those intervals describe this generator.

The street snapshot contains {{ corridor_segments }} Bloor Street segments, with frozen source checksums and longitude/latitude coordinates (EPSG:4326). No distances are computed from angular coordinates, and the simulation does not predict signal strength from the street geometry. {{ geography_attribution }} [Dataset]({{ geography_dataset_url }}); [licence]({{ geography_licence_url }}).

## Start with one incident

![A simulated commute, three signal-strength observations, and the connected network facts](figures/incident.png)

Consider a simulated commuter travelling along Bloor Street West. In this walkthrough, the phone's received signal strength changes from {{ first_phone_signal }} dBm before the disruption to {{ last_phone_signal }} dBm afterwards. At the middle observation, the connected backhaul link reports {{ middle_link_loss }}% packet loss, and peer phones using that dependency report {{ middle_peer_loss }}% mean loss.

The task is to find earlier episodes with a similar signal trajectory and network conditions, then show the engineer what they have in common.

The query is **{{ query_episode_id }}**, whose independently reconstructed pattern is **{{ query_pattern }}**. Its first retrieved comparison is **{{ candidate_episode_id }}**, labelled **{{ candidate_pattern }}**. The walkthrough uses the first final-test episode with an observed service disruption, selected before inspecting retrieval correctness. It illustrates the workflow; the experiments below evaluate all final-test queries.

A retrieved comparison is a lead for the engineer to inspect, not a confirmed cause of the dropped call.

{{ encoder_explanation }}

## Experiment 1: what do the operators preserve?

**Question.** Can the representation tell apart the same measurements attached to different meanings, appearing in a different order, or reached through a different network connection?

**Setup.** HDC needs only three operations: `bind(role, value)` keeps a measurement attached to its meaning, `bundle(facts)` combines contributions, and `permute(observation, position)` marks order. Learning reuses addition: `class_memory += normalize(episode_vector)`. Each test below builds two episodes that differ in exactly one way and compares their {{ dimension }}-dimensional hypervectors. A control repeats the test with the relevant operator removed. Cosine similarity of 1 means the two encodings are identical; lower values mean the representation tells them apart.

| What differs between the two episodes | Operator under test | Cosine with the operator | Without the operator |
|---|---|---:|---|
| A good state and a bad state, swapped between the phone's signal and an upstream network link | Binding | {{ bound_cosine }} | Identical encodings (max difference {{ unbound_error }}) |
| The same three observations, in reverse order | Permutation | {{ ordered_cosine }} | Identical encodings (max difference {{ unordered_error }}) |
| The same network measurements, with one serving edge rewired | Graph-joined context | {{ connected_cosine }} | Identical encodings (max difference {{ disconnected_error }}) |

Numeric levels behave as intended too: adjacent levels have cosine {{ adjacent_level_cosine }}, while distant levels have {{ distant_level_cosine }}, so nearby measurements receive similar encodings.

**Interpretation.** Each operator carries exactly the information it is meant to, and without it the distinction disappears entirely. Experiment 2 shows that these distinctions matter for retrieval. The pairs are controlled design checks, not a benchmark.

**Editing a query.** Because an episode is a sum of contributions, one fact can be removed by subtraction, with no need to re-encode the others. Removing the handset from the walkthrough query matches a complete re-encoding to within {{ handset_edit_error }}. {{ handset_edit_overlap }} A changed ranking answers the edited question; it is not automatically a better result.

## Experiment 2: can it retrieve useful earlier incidents, and explain them?

**Question.** Does HDC return earlier incidents with the same independently checked triage pattern, and can each result be traced to its source evidence?

**Setup.** For each final-test query, exact corridor and time filters first select the eligible earlier incidents in memory: {{ minimum_candidates }}–{{ maximum_candidates }} per query in the first run (mean {{ mean_candidates }}). Lance and an independently registered GeoDataFusion query agree on that selection. HDC then ranks the candidates by cosine similarity. A result counts as relevant when it has the query's triage pattern, which is not necessarily the same real-world cause. Precision@5 is the share of the top five results that are relevant; reciprocal rank@10 is 1 when the first result is relevant, 0.5 when the first relevant result is second, and so on.

![Retrieval precision with grouped uncertainty intervals](figures/retrieval.png)

| Method | Precision@5, 95% interval | Top-1 match | Reciprocal rank@10 |
|---|---:|---:|---:|
{{ retrieval_rows }}

**Result.** {{ retrieval_headline }}

**What order and connectivity contribute.** Removing the connected network context lowers precision@5 by {{ path_gain_mean }} points [{{ path_gain_lower }}, {{ path_gain_upper }}]; removing observation order lowers it by {{ order_gain_mean }} points [{{ order_gain_lower }}, {{ order_gain_upper }}]. These patterns are defined by how signal changes over time and by shared network dependencies, so large drops are expected. The ablations confirm that the encoder captures that structure, which a representation without order or connections cannot.

**Every result can be explained.** Each similarity score is a sum of contributions from individual encoded facts, such as the phone's signal at one observation or a link's packet loss. Each contribution links back to its source observations, telemetry and network edges. We checked this for all {{ top_hits_checked }} top results: the contributions reconstruct each float32 score to within {{ score_reconstruction_error }}, and a separately recorded residual accounts for float16 search storage. Contributions describe how a score was assembled, including overlap between facts; they are not causal explanations.

**Errors.** {{ retrieval_miss_summary }}

| Query | Expected pattern | Retrieved pattern |
|---|---|---|
{{ retrieval_error_rows }}

**Limit.** We did not run a non-HDC similarity search on the same measurements, so this experiment shows that HDC retrieves well, not that it retrieves better than the alternatives.

## Experiment 3: how quickly does class memory learn from reviews?

**Question.** How many reviewed incidents does each method need before it classifies new incidents well?

**Setup.** HDC starts with an empty memory for each of the four patterns. Each reviewed incident adds its hypervector to its pattern's memory, and the encoder never changes. LR and the MLP are trained on the same reviewed incidents, given as the same connected measurements in 21 numbers. Their settings are tuned for each review budget on separate validation worlds; [Reproduce and inspect](#reproduce-and-inspect) lists the details. All three methods are scored on the same independent final-test worlds using **macro F1**: the F1 score of each pattern, averaged with equal weight, where 100% is perfect classification.

![Learning from reviewed incidents with no encoder retraining](figures/learning.png)

{{ learning_headline }}

| Reviews per class | HDC | LR | MLP | HDC − LR, points | HDC − MLP, points |
|---:|---:|---:|---:|---:|---:|
{{ learning_rows }}

The first three columns show mean macro F1; the shaded bands in the figure show each method's own uncertainty. The last two columns compare methods directly: each test world scores all three, so we take the difference within each world, then give its mean with a 95% interval. **Bold** marks a difference whose interval excludes zero.

{{ hardest_pattern }}

**Interpretation.** HDC's advantage is learning from very few examples. A plausible reason: its class memory is a running sum over a fixed encoder, so one example per pattern already gives a usable comparison. LR and the MLP must estimate their weights from those same few examples. Once each pattern has a handful of reviews, the three methods perform about the same.

<details>
<summary>Details: per-pattern errors, delayed feedback and individual updates</summary>

**Errors by pattern.** At {{ error_budget }} reviews per class, this is the share of final-test incidents of each pattern assigned to the wrong pattern, averaged across the seed grid. These are descriptive rates without intervals; confusion matrices for every budget and world are retained.

| Actual pattern | HDC | LR | MLP |
|---|---:|---:|---:|
{{ class_error_rows }}

**Delayed feedback.** In this demonstration, a review becomes available {{ review_delay_s }} seconds after its incident's observation window ends. Replaying the memory-building episodes in time order, with each review added only once available, HDC triaged {{ delayed_accuracy }} of incidents correctly. It made a prediction for {{ delayed_coverage }} of them. The remaining incidents came before any review had arrived; for those it reported “insufficient labelled memory” rather than guess, and they count as incorrect. This replay uses memory-building episodes, not the independent final-test worlds above.

**A single update can help or hurt.** Examples from the first replay, measured on a fixed probe set inside the memory partition, are listed below. They illustrate update effects; they are not held-out estimates.

{{ update_effects }}

Every update is logged with its review provenance, availability time and before/after memory hashes, and can be reversed exactly by restoring the previous memory snapshot.

**Fit diagnostics.** There are {{ classifier_warning_count }} convergence warnings among the {{ classifier_fit_count }} final LR/MLP fits, and {{ selection_warning_count }} among {{ selection_fit_count }} validation-selection fits. Diagnostics and iteration counts are retained for every fit.

</details>

## Experiment 4: what do prediction, learning and storage cost?

**Question.** How long does each method take to triage one incident, and to learn from newly reviewed incidents? How much storage does each representation need?

**How to read the timings.** Every time in this section is the elapsed wall-clock time of one complete call: nothing is divided by a batch size or averaged across reviews. Each call is repeated on one CPU thread with warm caches; we take its median within each of the {{ pair_count }} seed pairs, then report the median across pairs. The 95th percentile stays within {{ p95_overhead }}% of the median for every reported operation, so medians are representative; all p95 values are retained in `metrics.json`. Graph joins, geographic filtering and database work are excluded. Measurements come from one machine ({{ runtime_platform }}, Python {{ runtime_python }}), so compare them with each other rather than treating them as absolute serving latency.

### Predicting one incident

| Method | What one prediction involves | Time, ms |
|---|---|---:|
{{ prediction_rows }}

{{ prediction_finding }}

{{ learning_update_explanation }}

### Storage

| Item | HDC | LR | MLP |
|---|---:|---:|---:|
| One encoded incident | {{ raw_vector_bytes }} B (float32); {{ search_vector_bytes }} B float16 search copy | {{ model_input_bytes }} B | {{ model_input_bytes }} B |
| Learned state at {{ max_reviews_per_class }} reviews per class | {{ class_memory_bytes }} B (four class accumulators) | {{ lr_model_bytes }} B | {{ mlp_model_bytes }} B |
| Earlier reviews kept for the next update | none | {{ review_buffer_bytes }} B | {{ review_buffer_bytes }} B |

HDC needs far more storage per incident than the 21-number LR/MLP input; this study shows no compression benefit. In exchange, HDC's learned state is updated in place, while LR and the MLP must keep earlier reviews for retraining. Serialized LR/MLP sizes include estimator metadata and any fitted scaler; HDC's class-memory size excludes audit history, exact-undo snapshots and the encoder basis.

<details>
<summary>Supporting measurements: initial fits, pipeline stages and artifact sizes</summary>

**Initial fit from scratch.** Median time to build each learner from its reviewed examples at every budget, once per run, excluding encoding and persistence. Model-selection compute is recorded separately in the frozen settings.

| Reviews per class | HDC memory building, ms | LR fitting, ms | MLP fitting, ms |
|---|---:|---:|---:|
{{ fit_rows }}

**Pipeline stages in the first run.** Encoding all episodes takes {{ encode_seconds }} s, persisting vectors and manifests {{ persistence_seconds }} s, and all validation/test geography gates plus independent checks {{ eligibility_seconds }} s. Median LanceDB retrieval is {{ lance_retrieval_ms }} ms; the matched in-memory HDC scoring-and-ranking median is {{ memory_retrieval_ms }} ms. These measure different layers and are not a speed contest.

**Artifact sizes in the first run.** Directory totals include retained Lance versions and metadata. Tensor totals exclude Python objects, source tables, temporary allocations and audit history; peak process memory was not measured.

| Resource | Size |
|---|---:|
| All float32 HDC episode tensors | {{ representation_mib }} MiB |
| All LR/MLP input tensors | {{ model_input_mib }} MiB |
| Cached encoder basis tensors | {{ encoder_cache_mib }} MiB |
| Lance vector store, raw/search vectors and manifests | {{ vector_store_mib }} MiB |
| Lance source-record store | {{ source_store_mib }} MiB |

</details>

**Limits.** Incremental optimizers and warm starts could lower LR/MLP update cost; they were not evaluated, so these results do not show that conventional models must retrain from scratch. Near-real-time suitability also depends on the complete data path, which this study does not time.

## Conclusions

| Claim | Evidence | Boundary |
|---|---|---|
{{ claims_rows }}

The simulator uses simple observed rules and deliberately balanced classes, with complete telemetry. This makes the mechanisms observable and supplies a compact feature table that conventional trained classifiers can use effectively. It does not demonstrate operational diagnosis, realistic class prevalence, adaptation under drift, robustness to missing telemetry, or coverage of new incident types.

{{ geographic_enhancements }}

{{ future_work }}

## Reproduce and inspect

Inspect the [source code]({{ source_link }}) for the implementation. Run `uv run src/prepare.py`, `uv run src/run.py`, then `uv run src/report.py` from the repository root. All three scripts use the paths in `src/settings.py`. The manifest records dependency versions, source checksums, configuration, encoder and code hashes, and all completed seed pairs. The committed metrics retain block-level results, per-class errors and repeated timing measurements. Reproducing a run generates the detailed predictions, review logs, retrieval errors, query edits and source witnesses locally.

The [LR](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) implementations come from locked scikit-learn 1.9.1. The encoder weighting was selected on separate validation worlds (data seeds 1001–1005) and frozen before this evaluation. This run uses fresh data seeds {{ data_seeds }}: new independent simulated worlds, not external carrier validation. LR and the MLP receive 21 inputs: six connected measurements at each of three ordered observations, plus three handset-category indicators. Both are optimized with L-BFGS to a declared tolerance, not limited to one training pass. Their settings are selected separately at each review budget, using only this run's first data seed's validation worlds and averaging across three initialization seeds, then frozen before final testing.

The declared search space uses LR C ∈ {0.1, 1, 10}, MLP hidden width ∈ {16, 32}, MLP alpha ∈ {0.001, 0.1}, and fixed physical range scaling or an additional training-fitted StandardScaler. C controls inverse L2 regularization strength; alpha controls MLP regularization. The HDC weighting is fixed throughout this run; no HDC variant search, additional feature engineering or test-based selection is performed. Only the active encoder is evaluated, alongside operator ablations that answer the representation questions. The frozen choices are:

| Reviews/class | LR C | LR scaling | MLP width | MLP alpha | MLP scaling |
|---|---:|---|---:|---:|---|
{{ selected_settings_rows }}
