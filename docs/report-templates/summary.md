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

**Hyperdimensional computing (HDC)** represents these facts in long numeric arrays called hypervectors. Its operations can attach a value to a role, combine contributions, and preserve order. We study whether an episode built with those operations can support retrieval, inspection of its source evidence, and learning through additions to labelled class memory while the encoder stays fixed.

The experiments examine four questions:

- **Representation:** Do roles, event order and network connectivity change the meaning of an encoded incident?
- **Retrieval and evidence:** Can it find comparable earlier incidents and return the source records supporting the comparison?
- **Learning:** How useful does class memory become as reviewed examples arrive, and when do updates help or hurt?
- **Resources:** What do encoding, prediction, learning updates, retrieval and storage cost, including the cost per new review?

The learning comparison evaluates HDC against regularized logistic regression and a small MLP. All three learners receive the same reviewed incidents and connected measurements. Retrieval evaluates HDC itself, with order and connectivity ablations as internal representation checks. These checks are not additional comparison models.

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

An independent checker reconstructs these labels from source measurements, event order and valid edges. Encoder input does not include generator scenario names, labels or service-disruption outcomes. Review labels become available {{ review_delay_s }} seconds after the complete episode; they stand in for analyst feedback in the demonstration. No human reviews were collected.

Each data seed produces {{ episode_count }} episodes across {{ block_count }} independent network/time blocks. Of those blocks, {{ memory_blocks }} build labelled memory, {{ validation_blocks }} support model selection, and {{ test_blocks }} are reserved for final testing. Subscribers and shared incidents stay within their assigned block. No validation or final-test label updates memory.

The full experiment crosses {{ data_seed_count }} data seeds with {{ encoder_seed_count }} encoder seeds. It scores {{ final_query_count }} final queries across repetitions, representing {{ final_world_count }} distinct final-test worlds. Repeated encoders reuse episodes and are averaged before uncertainty intervals resample data seeds and complete worlds. Those intervals describe this generator.

The street snapshot contains {{ corridor_segments }} Bloor Street segments, with frozen source checksums and longitude/latitude coordinates (EPSG:4326). No distances are computed from angular coordinates, and the simulation does not predict signal strength from the street geometry. {{ geography_attribution }} [Dataset]({{ geography_dataset_url }}); [licence]({{ geography_licence_url }}).

## Start with one incident

![A simulated commute, three signal-strength observations, and the connected network facts](figures/incident.png)

Consider a simulated commuter travelling along Bloor Street West. In this walkthrough, the phone's received signal strength changes from {{ first_phone_signal }} dBm before the disruption to {{ last_phone_signal }} dBm afterwards. More negative dBm values indicate weaker received signal strength. At the middle observation, the connected backhaul link reports {{ middle_link_loss }}% packet loss, and peer phones using that dependency report {{ middle_peer_loss }}% mean loss.

These facts make the comparison specific: look for an earlier episode with a similar signal-strength trajectory and connected network conditions. Exact spatial and time filters first identify eligible earlier memory episodes. Similarity then ranks them, and retained observations, telemetry and valid edges let the engineer inspect what the retrieved episode has in common.

The query is **{{ query_episode_id }}**, whose independently reconstructed pattern is **{{ query_pattern }}**. Its first retrieved comparison is **{{ candidate_episode_id }}**, labelled **{{ candidate_pattern }}**. The walkthrough uses the first final-test episode with an observed service disruption, selected before inspecting retrieval correctness. It illustrates the workflow; the experiments below evaluate all final-test queries.

A retrieved comparison is evidence for further inspection. The triage labels describe the observed patterns in this simulation; they do not confirm the cause of the commuter's dropped call. The next section shows how the phone and network facts become one composable representation, before we measure retrieval and learning performance.

{{ encoder_explanation }}

## Experiment 1: what do the operators preserve?

**Question.** Can the representation distinguish the same values attached to different meanings, appearing in different orders, or connected through different edges?

**Setup.** Binding attaches a value to a role. Bundling adds contributions. Permutation marks an observation's place in the episode. The phone signal channel has weight {{ phone_signal_weight }}; connected cell/link/peer context has weight {{ context_weight }}. Numeric levels preserve neighbourhoods; the handset contribution has weight {{ handset_weight }}. The dimension is {{ dimension }}.

Hyperdimensional computing (HDC) represents information in long numeric arrays called hypervectors. Here, each atomic role starts as a seeded array of +1 and −1 values. Distinct roles have little overlap, while nearby numeric measurements deliberately receive correlated arrays. An episode becomes a sum of these encoded contributions rather than an opaque identifier.

In this implementation, binding multiplies arrays element by element, bundling adds them, and permutation rotates their coordinates by a fixed number of positions. Binding gives radio strength a different meaning from link loss. A position-specific rotation distinguishes the same observations in reverse order. The connected channel follows the time-valid phone → cell → backhaul dependency before encoding its measurements and peer context.

The operator vocabulary is small: `bind(role, value)` keeps a measurement attached to its meaning; `bundle(facts)` combines contributions; `permute(observation, position)` marks order. Learning reuses addition: `class_memory += normalize(episode_vector)`. The encoder implements these operations with TorchHD; no text embedding model is needed for these tabular records.

**Result.** Swapping good/bad states between phone radio and an upstream link gives cosine {{ bound_cosine }}. Omitting binding makes the two accumulators equal within {{ unbound_error }}. Reversing the observations gives cosine {{ ordered_cosine }}; omitting order leaves error {{ unordered_error }}. Rewiring one serving edge gives cosine {{ connected_cosine }}; pooling the same network measurements without their connections leaves error {{ disconnected_error }}. The adjacent numeric-level cosine is {{ adjacent_level_cosine }}, compared with {{ distant_level_cosine }} for distant levels.

**Interpretation.** These controlled illustrations show exactly what the operators do. The graph pair retains every measured value and changes one edge in a separate counterfactual world. Its two-candidate ranking has a 50% random reference and serves as an illustration, not a practical graph benchmark. Broader retrieval usefulness is measured next.

The handset contribution can be removed from a raw query without re-encoding the other facts. Reconstruction error in the walkthrough is {{ handset_edit_error }}. The original top five are `{{ original_top_five }}`; after removing handset they are `{{ edited_top_five }}`. A changed ranking is an editable query, not automatically a better result.

## Experiment 2: can it retrieve useful earlier incidents?

**Question.** Does the representation return earlier incidents with the same independently checked triage pattern, and can their source evidence be inspected?

**Setup.** Every query uses an exact corridor/time filter and a memory-only candidate set. All three observations must qualify. Lance and independently registered GeoDataFusion agree on selected observation identities. The first run has {{ minimum_candidates }}–{{ maximum_candidates }} candidates per query (mean {{ mean_candidates }}). Relevance means the same triage pattern, not the same real-world cause. Connectivity and order are removed separately in HDC ablations to check their contribution to retrieval.

![Retrieval precision with grouped uncertainty intervals](figures/retrieval.png)

| Method | Precision@5, 95% interval | Top-1 match | Reciprocal rank@10 |
|---|---:|---:|---:|
{{ retrieval_rows }}

Random ranking yields expected precision {{ random_precision }}, based on each query's eligible label prevalence. The paired HDC difference when connectivity is omitted is {{ path_gain_mean }} percentage points [{{ path_gain_lower }}, {{ path_gain_upper }}]. When order is omitted, the difference is {{ order_gain_mean }} points [{{ order_gain_lower }}, {{ order_gain_upper }}].

**Evidence.** Every stored-vector top hit was checked against its source records. Maximum raw reconstruction error across runs is {{ reconstruction_error }}; maximum float32 reference-score reconstruction error is {{ score_reconstruction_error }}. Reference scores are decomposed into additive term contributions under the raw accumulator's normalization. A separately recorded quantization residual connects that reference to the stored float16 vector's cosine. These contributions include cross-term interference; they are not causal importance scores. Source identities and timestamps support each term.

**Errors.** The first fixed run has {{ retrieval_mismatches }} top-1 mismatches out of {{ query_count }}. Examples are retained rather than discarded:

| Query | Expected pattern | Retrieved pattern |
|---|---|---|
{{ retrieval_error_rows }}

**Interpretation.** The HDC ablations reveal the value of retaining order and connected context. This retrieval experiment does not establish an advantage over LR or MLP, which are evaluated as classifiers in Experiment 3. Float16 search storage is independently verified against its quantized vectors; its mean precision difference from float32 is {{ quantization_precision_change }} points. Float32 accumulators remain authoritative for arithmetic and updates.

## Experiment 3: how does memory grow through reviews?

**Question.** How useful is class memory with few labelled incidents, and what happens when feedback arrives after a decision?

**Setup.** Each class starts with no vector. Encoding produces an episode hypervector; a review adds its normalized vector to the appropriate float32 class accumulator. Prediction compares the episode with normalized class memories. The comparison models are regularized multinomial logistic regression and a small one-hidden-layer ReLU MLP. Both receive the same 21 input features: six measurements at each of three ordered observations, plus three handset-category indicators. The connected measurements come from the same valid graph joins as HDC. Validation chooses between the already declared physical range scaling and an additional StandardScaler fitted only to the reviewed memory examples at each budget. Both models use L-BFGS optimization to a declared tolerance; they are not limited to one training pass. LR regularization and MLP width/regularization are selected separately at each budget using the first data seed's validation worlds, averaged across three initialization seeds, then frozen for every final test. Every method receives the same reviewed incidents. Review budgets count training labels; additional labelled validation worlds support model selection.

![Learning from reviewed incidents with no encoder retraining](figures/learning.png)

| Reviews per class | HDC macro F1 | Trained LR macro F1 | Small MLP macro F1 |
|---:|---:|---:|---:|
{{ learning_rows }}

**Result.** {{ learning_direction }} {{ low_sample_finding }} The curve can plateau or regress at intermediate budgets; more reviews do not guarantee improvement. Macro F1 gives each pattern equal weight, with 1.0 representing perfect classification. The first delayed-feedback replay's HDC accuracy is {{ delayed_accuracy }}, with prediction coverage {{ delayed_coverage }}. Coverage includes initial decisions for which no review has arrived; these produce “insufficient labelled memory”. This replay uses memory-building episodes, while the learning curves above use independent final-test worlds.

At {{ error_budget }} reviews per class, the paired, world-grouped differences are {{ learning_difference }}. There are {{ classifier_warning_count }} convergence warnings among the {{ classifier_fit_count }} final LR/MLP fits, and {{ selection_warning_count }} among {{ selection_fit_count }} validation-selection fits. Diagnostics and iteration counts are retained for every fit.

The aggregate score can hide a difficult pattern. At {{ error_budget }} reviews per class, the following share of final-test examples is assigned to the wrong class, averaged across the seed grid. These are descriptive per-class errors; the grouped uncertainty intervals above apply to macro F1. Detailed confusion matrices are retained for every budget and world.

| Actual pattern | HDC error rate | Trained LR error rate | Small MLP error rate |
|---|---:|---:|---:|
{{ class_error_rows }}

Updates are not guaranteed to help. The first fixed replay contains these examples:

{{ update_effects }}

These examples use a fixed diagnostic probe within the memory partition, including episodes that are later reviewed. They are neither held-out performance estimates nor a signal for choosing updates. All updates follow the configured review schedule. The full log records review provenance, availability time, encoder hash, update-vector hash and before/after accumulator hashes. Exact reversal restores the previous float32 snapshot rather than relying on rounded subtraction.

**Interpretation.** There is supervised learning, but no encoder retraining or optimization loop for the HDC memory. Adding unlabelled records to a retrieval store is a different operation. The learning curves determine how strongly we can describe sample efficiency on these patterns; they do not establish it across telecom tasks.

## Experiment 4: how cheap is the complete operation?

**Question.** What does it cost to encode an episode, score it and update memory, and how much storage is used?

**Setup.** One Torch/BLAS CPU thread; batch one; {{ warmup }} warm-up iterations and {{ benchmark_repeats }} measurements for prediction/addition per seed pair. More expensive classifier refits use {{ refit_warmup }} warm-ups and {{ refit_repeats }} measurements. The table reports the median of run medians and the median of run p95 measurements. The runtime is {{ runtime_platform }}, Python {{ runtime_python }}. Joins, geo selection and database writes are separate from these warm compute measurements.

![Measured CPU operation costs, with addition distinguished from the full operation](figures/compute.png)

| Operation | Median of run medians | Median of run p95 |
|---|---:|---:|
{{ timing_rows }}

HDC's raw accumulator uses {{ raw_vector_bytes }} bytes per episode; its search vector uses {{ search_vector_bytes }} bytes in float16. The LR/MLP input vector uses {{ model_input_bytes }} bytes in float32. Four HDC class accumulators use {{ class_memory_bytes }} bytes, before mappings, counters, audit history and exact-undo snapshots. These sizes describe representations, not the complete trained models. This dataset supports no compression claim.

Measured resource totals in the first fixed run:

| Resource | Size |
|---|---:|
| All float32 HDC episode tensors | {{ representation_mib }} MiB |
| All LR/MLP input tensors | {{ model_input_mib }} MiB |
| Cached encoder basis tensors | {{ encoder_cache_mib }} MiB |
| Lance vector store, raw/search vectors and manifests | {{ vector_store_mib }} MiB |
| Lance source-record store | {{ source_store_mib }} MiB |

LR and MLP trained estimators, including any fitted scalers, are saved and checked for identical predictions after reloading. Their serialized sizes and initial fit times are recorded separately at every review budget.

At {{ max_reviews_per_class }} reviews per class, median serialized estimator size is {{ lr_model_bytes }} bytes for LR and {{ mlp_model_bytes }} bytes for MLP. These include estimator metadata and any fitted scaling, but exclude the retained review buffer. That buffer uses {{ review_buffer_bytes }} bytes for the refit measurement. HDC's class tensor size also excludes audit history and encoder basis tensors.

Observed median initial fit times across the seed grid, excluding encoding and persistence:

| Reviews per class | HDC memory building, ms | LR fitting, ms | MLP fitting, ms |
|---|---:|---:|---:|
{{ fit_rows }}

These initial fits are recorded once per run and budget; the repeated refit benchmark above supplies a separate measurement of incorporating a further review. Model-selection compute is recorded in the frozen settings and is not included in these initial fits.

The directory totals include retained Lance versions and metadata. They describe these artifacts rather than an optimized storage comparison. Tensor totals exclude Python objects, source tables, temporary allocations and audit history; peak process memory was not measured.

The first run encodes all episodes in {{ encode_seconds }} s, persists the vectors and manifests in {{ persistence_seconds }} s, and performs all validation/test geography gates plus independent checks in {{ eligibility_seconds }} s. LanceDB retrieval median is {{ lance_retrieval_ms }} ms; the matched in-memory HDC scoring-and-ranking median is {{ memory_retrieval_ms }} ms. They measure different layers and are not an implementation speed contest.

**Interpretation.** The addition is cheap. For LR and MLP, review incorporation here means encoding the new incident, refitting on {{ refit_training_examples }} retained reviewed examples, then predicting it. HDC encodes, predicts with its existing class memory, and adds the new review. Those are different update strategies. Conventional incremental optimizers and warm starts could reduce refit cost; they are not evaluated here, so this comparison supports no general claim that conventional ML must retrain from scratch. The complete operation and alternative methods deserve equal attention. These are component measurements on one machine, without energy, production serving, spatial-index or ANN benchmarks. Near-real-time suitability requires an application latency budget and its complete data path.

{{ learning_update_explanation }}

## What this supports

| Claim | Evidence | Boundary |
|---|---|---|
{{ claims_rows }}

The simulator uses simple observed rules and deliberately balanced classes, with complete telemetry. This makes the mechanisms observable and supplies a compact feature table that conventional trained classifiers can use effectively. It does not demonstrate operational diagnosis, realistic class prevalence, continual adaptation under drift, robustness to missing telemetry, or coverage across new incident types. Those would need separate experiments.

{{ geographic_enhancements }}

## Takeaways

In this controlled study, HDC combines role-sensitive, ordered representations with connected network evidence. Retrieval returns comparable earlier incidents with source records that can be inspected, and reviewed examples improve class memory through reversible additions while the encoder stays fixed.

HDC performs well with few reviewed examples. Its advantage is clearest at the smallest label budgets; LR and the MLP become competitive as more reviews arrive. The results support composable representation and incremental memory building, with usefulness measured against familiar trained classifiers.

The resource comparison depends on the update strategy and batch size. HDC incorporates individual reviews quickly, while LR batch retraining amortizes well and is cheaper per review at the largest measured batch. HDC uses more representation storage than the LR/MLP input vector. These findings apply to the simulated patterns and measured CPU workload; operational telemetry, missing data and drift remain untested.

## Reproduce and inspect

Inspect the [source code]({{ source_link }}) for the implementation. Run `uv run src/prepare.py`, `uv run src/run.py`, then `uv run src/report.py` from the repository root. All three scripts use the paths in `src/settings.py`. The manifest records dependency versions, source checksums, configuration, encoder and code hashes, and all completed seed pairs. The committed metrics retain block-level results, per-class errors and repeated timing measurements. Reproducing a run generates the detailed predictions, review logs, retrieval errors, query edits and source witnesses locally. The active learning comparison uses trained LR and a small MLP.

The [LR](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) implementations come from locked scikit-learn 1.9.1. The encoder weighting was selected on separate validation worlds (data seeds 1001–1005) and frozen before this evaluation. The full study defaults now use fresh data seeds 1006–1010. These are new independent simulated worlds, not external carrier validation. The actual data seeds in this run are {{ data_seeds }}. LR/MLP settings use only this run's first data seed's validation worlds and are frozen before final testing.

The declared search space uses LR C ∈ {0.1, 1, 10}, MLP hidden width ∈ {16, 32}, MLP alpha ∈ {0.001, 0.1}, and fixed physical range scaling or an additional training-fitted StandardScaler. C controls inverse L2 regularization strength; alpha controls MLP regularization. The HDC weighting is fixed throughout this run; no HDC variant search, additional feature engineering or test-based selection is performed. Only the active encoder is evaluated, alongside operator ablations that answer the representation questions. The frozen choices are:

| Reviews/class | LR C | LR scaling | MLP width | MLP alpha | MLP scaling |
|---|---:|---|---:|---:|---|
{{ selected_settings_rows }}
