# Walkthrough: connected telecom records, hypervectors and class prototypes

This walkthrough explains the representation and prototype learner in this codebase. It assumes no telecom knowledge. It follows one generated incident through value bucketing, role-value binding, structural permutation, observation-order permutation, weighting and bundling. The final section explains how the same represented episodes become class prototypes through supervised learning.

Related figures are available under `runs/reports/figures/`.

## 1. First, what is happening in the telecom example?

Imagine someone travelling with their phone. The phone connects wirelessly to the mobile network. That connection depends on several things:

- **Subscriber:** the customer.
- **Handset:** another word for the phone. Here, “handset model” means its product category, such as `model_0`.
- **Radio connection:** the wireless connection between the phone and network equipment.
- **Serving cell:** the part of the network providing that phone’s wireless connection at that moment. As the person moves, a different cell may serve them.
- **Backhaul link:** the connection carrying traffic from that cell onward into the operator’s network.
- **Peer phones:** other phones whose traffic uses the same backhaul link. They can share that link even when different cells serve them.

Our phone therefore depends on this chain:

**Phone → serving cell → backhaul link**

If other phones share the link, their measurements give us additional context about that shared dependency.

Everything except the public Toronto street geometry is simulated.

## 2. The graph is assembled from records before HDC begins

The code stores separate tables rather than one preassembled graph:

| Records | What they contain |
|---|---|
| `subscribers`, `phones` | Who owns the phone and which handset model it is |
| `cells`, `links` | Network components |
| `edges` | Which cell uses which link, including when that connection is valid |
| `observations` | Our phone’s measurements, serving cell and observation time |
| `cell_status`, `link_status` | Measurements of network components at particular times |
| `peer_observations` | Measurements associated with other phones’ cells and connections |
| `episodes` | The three observations belonging to one incident |
| `reviews` | Later labels; these do not enter the representation |

An **episode** is our unit of representation: three observations at **0, 20 and 40 seconds**.

At each observation, the dataset builder follows the phone’s serving cell, finds that cell’s time-valid backhaul link, and collects measurements from those components. It also averages the peer measurements associated with that same link at that same time.

This connection work matters. We want facts about **the network our phone was using**, rather than arbitrary measurements from elsewhere.

The source records have database properties such as `radio_dbm`, `cell_load_pct` and `link_loss_pct`. Their property-value assignments become **role-value pairs** when we bind their hypervector representations.

## 3. Now, meet the six measurements

Our running episode is `S1006-W18-E00-0`. Its phone is `model_0`. The values below are rounded for display; the encoder buckets the original measurements. The worked hypervectors use encoder seed 2001 and dimension 4,096.

| Measurement | Meaning | 0 seconds | 20 seconds | 40 seconds |
|---|---|---:|---:|---:|
| Phone signal strength | How much wireless signal power reaches our phone | −80.60 dBm | −91.74 dBm | −110.64 dBm |
| Serving cell load | How busy the cell is, using a simulated percentage | 44.17% | 25.84% | 40.56% |
| Backhaul packet loss | Percentage of data pieces lost on the link | 1.24% | 0.95% | 0.26% |
| Backhaul latency | Delay on that link | 18.19 ms | 14.67 ms | 16.10 ms |
| Peer mean signal strength | Average wireless signal strength for the selected peers | −92.15 dBm | −81.89 dBm | −89.14 dBm |
| Peer mean packet loss | Average loss reported by those peers | 1.24% | 0.95% | 0.25% |

Two units need unpacking:

- **dBm** measures signal power on a logarithmic scale. For this example, remember that **−80 dBm is stronger than −110 dBm**. Our phone’s signal is getting weaker.
- **ms** means milliseconds: one thousandth of a second.

A **packet** is a small piece of transmitted data. Packet loss measures how many of those pieces fail to arrive. **Latency** measures delay.

## 4. Three components, nineteen contributions

The encoder organizes these facts into three components:

| Component | Contents | Number of contributions |
|---|---|---:|
| Phone signal | Our phone’s signal strength at three times | 3 |
| Connected network context | Cell load, link loss, link delay and two peer averages at three times | 15 |
| Handset model | The phone model, once | 1 |

That gives us **19 contributions to one hypervector**.

“Context” is one component built by bundling 15 measurement hypervectors: five measurements at each of three observation times.

At each time, we collect cell load, backhaul packet loss, backhaul delay, peer average signal strength and peer average packet loss. Each measurement becomes **one bound hypervector term**. We repeat that at 0, 20 and 40 seconds.

We repeat the measurements because the network’s condition can change during the episode. The observation-order permutation tags each term with its place in that sequence.

**We are counting measurements, not network objects.** Even if the same link serves the phone throughout, its loss and delay at three times produce six terms.

And each peer **average** contributes one term—we don’t encode every peer separately.

**Phone signal contributes one measurement per observation, so it contributes three terms across the whole episode.** Phone signal describes the commuter phone’s own wireless signal strength. Context describes the connected network and other phones sharing its connection.

| At each observation time | Phone signal | Context |
|---|---|---|
| Measurements | Our phone’s signal strength | Cell load, link loss, link delay, peer average signal strength, peer average loss |
| Terms per time | **1** | **5** |
| Observation times | 3 | 3 |
| Total terms per episode | **3** | **15** |

**That asymmetry is a deliberate choice in this demo’s dataset and encoder. HDC doesn’t require it.** Phone signal and context classify facts by whose condition they describe. They don’t promise equally sized descriptions. Five is a selected feature set, not a property of connected data or HDC.

The **handset model** is the component that contributes only **one term for the entire episode**, because the phone model stays constant.

## 5. Before the incident: what are the building blocks?

**Hyperdimensional computing (HDC)** represents these facts in long numeric arrays called **hypervectors**. Its operations can attach a value to a role, combine contributions, and preserve order.

The encoder has two kinds of reusable hypervectors:

1. **Seeded bipolar role and category hypervectors**, containing +1 and −1, for meanings such as “phone,” “signal strength,” “serving relationship” and “phone signal channel.” Their names and encoder seed determine them reproducibly.
2. **32 correlated numeric level hypervectors**, numbered 0–31. Nearby levels are similar, so nearby measurements can remain similar.

There is one shared numeric level family. Role-value binding distinguishes, for example, a signal-strength level from a packet-loss level.

Every hypervector is written as $h$ with a descriptive suffix: for example, $h_{\text{phone}}$, $h_{\text{radio property}}$ or $h_{\text{level},23}$. Scalars such as measurements, indices and weights are not hypervectors and do not receive this notation.

| Notation | Meaning |
|---|---|
| $h_{\text{phone}}$ | Hypervector representing the phone’s object role |
| $h_{\text{radio property}}$ | Hypervector representing the signal-strength property role |
| $h_{\text{level},q}$ | Hypervector representing numeric bucket $q$ |
| $h_{\text{phone signal channel}}$ | Hypervector marking the phone signal channel |
| $\otimes$ | Binding |
| $\oplus$ | Bundling |
| $\rho^k(h)$ | Rotate hypervector $h$ right by $k$ coordinates, wrapping around |

## 6. Organize numerical values into buckets

The encoder uses **bucketing**: organize numerical values into bins over a fixed range, then use the corresponding level hypervector.

For a concrete measurement, the encoder:

1. Maps it into its field’s fixed range.
2. Clips that fraction to 0–1.
3. Multiplies by 31 and rounds to select a bucket numbered 0–31.
4. Selects that bucket’s level hypervector.

For a measurement $x$ of property $f$, with chosen bounds $a_f$ and $b_f$:

$$
s_f(x)=\mathrm{clip}\!\left(\frac{x-a_f}{b_f-a_f},0,1\right),
\qquad
q_f(x)=\mathrm{round}\!\left((K-1)s_f(x)\right),
\qquad K=32.
$$

Here $s_f(x)$ is a scalar fraction, $q_f(x)$ is a scalar bucket index, and $h_{\text{level},q_f(x)}$ is the selected hypervector.

This is ordinary linear range scaling followed by rounding into buckets. It is a representation choice, not a telecom-specific equation. The assumption is that equal steps in the measurement’s chosen units should produce equal steps along the encoding scale.

### Our first phone measurement

**We aren’t calculating the phone’s measurement**—that measurement is already recorded. We’re calculating where it falls within the encoder’s chosen range, so we can assign it to a bucket.

The first recorded signal strength is approximately **−80.603 dBm**. Remember: less negative means stronger signal.

| Symbol | Value | Meaning |
|---|---:|---|
| $x$ | −80.603 dBm | The measurement we want to encode |
| $a_f$ | −125 dBm | The lower, weaker-signal end of the encoding range |
| $b_f$ | −65 dBm | The upper, stronger-signal end of the encoding range |

These bounds are configuration choices. They are **not** the weakest and strongest observations in this particular episode.

**The numerator, $x-a_f$, asks: “How far above the lower bound is this measurement?”**

$$
x-a_f=-80.603-(-125)=44.397.
$$

Subtracting a negative adds its magnitude. So our measurement sits **44.397 units along the scale from its lower end**.

**The denominator, $b_f-a_f$, asks: “How wide is the entire encoding range?”**

$$
b_f-a_f=-65-(-125)=60.
$$

The denominator is the full width of the chosen interval.

**Dividing tells us what fraction of that interval we have travelled.**

$$
s_f(x)=\frac{44.397}{60}\approx0.740.
$$

So the measurement is approximately **74% of the way from the weaker bound to the stronger bound**.

That is a position on the chosen **dBm scale**. It does **not** mean “74% signal power” or “74% connection quality.”

**Only then do we choose a numeric bucket and its level hypervector.** There are 32 levels, indexed **0 through 31**. We map our fraction onto those indices:

$$
q_f(x)=\mathrm{round}(31\times0.740)
=\mathrm{round}(22.94)=23.
$$

We therefore select $h_{\text{level},23}$, the level-23 hypervector. Our three phone measurements become buckets **23, 17 and 7**.

The sequence is:

**Recorded measurement → position within a fixed range → nearest bucket index → corresponding hypervector.**

### The chosen bounds and their consequences

| Property | Encoding range | Practical interpretation of the choice |
|---|---|---|
| Phone and peer signal strength | −125 to −65 dBm | A shared working range covering generated phone signals of roughly −116 to −75 dBm and peer signals of roughly −94 to −74 dBm, with headroom |
| Cell load | 0–100% | The full percentage scale; generated loads cover only part of it |
| Link and peer packet loss | 0–10% | Focuses the buckets on the simulated low-loss regime rather than the entire possible 0–100% interval |
| Link delay | 0–120 ms | A nonnegative working range covering generated delays of 8–100 ms, with headroom |

These are modelling choices. The fixed bounds are not statistics fitted to the final dataset, and they do not calibrate connection quality.

For signal strength, the 60 dB interval and 32 level centres give a spacing of $60/31\approx1.94$ dB. Wider bounds reduce resolution; narrower bounds increase clipping. Clipping makes all values beyond an endpoint equivalent, and rounding can make nearby measurements equivalent.

The same level family serves every numeric property. Binding each value to its property role distinguishes signal strength from packet loss, even if their scaled fractions coincide.

The phone model is a category rather than a numerical measurement, so it gets a category hypervector rather than a numeric bucket. `model_2` does not mean a numerically larger phone than `model_1`.

## 7. The three operations are straightforward; their placement is the complicated part

We use $\otimes$ for **binding**, $\oplus$ for **bundling**, and $\rho^k$ for rotating right by $k$ coordinates.

| Operation | Arithmetic here | Purpose |
|---|---|---|
| Binding | Multiply corresponding coordinates | Attach a value to its property role and connected meaning |
| Bundling | Accumulate corresponding coordinates without thresholding | Combine hypervector contributions |
| Permutation | Rotate coordinates, wrapping around | Mark structural position or observation order |

For a tiny illustration, let:

$$
h_{\text{role}}=[1,-1,1,-1],
\qquad
h_{\text{value}}=[1,1,-1,-1].
$$

Then:

$$
h_{\text{role}}\otimes h_{\text{value}}=[1,-1,-1,1],
$$

$$
h_{\text{role}}\oplus h_{\text{value}}=[2,0,0,-2],
$$

$$
\rho^1(h_{\text{role}})=[-1,1,-1,1].
$$

These short arrays illustrate arithmetic only; the implementation uses 4,096-dimensional arrays.

There is **no sign threshold after bundling**. The encoder retains the weighted numeric bundle.

## 8. Let’s build the phone signal component first

Take the initial signal measurement, represented by $h_{\text{level},23}$.

The database property-value assignment is `radio_dbm = -80.603…`. For binding, its **role-value pair** is the signal-strength property role and the hypervector for bucket 23. The phone role and channel marker add the meaning of whose property it is and which component it belongs to.

Bind together:

$$
h_{\text{phone signal fact},0}
=h_{\text{phone signal channel}}
\otimes h_{\text{phone}}
\otimes h_{\text{radio property}}
\otimes h_{\text{level},23}.
$$

Read that as:

> “In the phone signal component, the commuter phone’s signal-strength measurement has numeric level 23.”

The later terms follow the same recipe, using levels 17 and 7:

$$
h_{\text{phone signal fact},1}
=h_{\text{phone signal channel}}
\otimes h_{\text{phone}}
\otimes h_{\text{radio property}}
\otimes h_{\text{level},17},
$$

$$
h_{\text{phone signal fact},2}
=h_{\text{phone signal channel}}
\otimes h_{\text{phone}}
\otimes h_{\text{radio property}}
\otimes h_{\text{level},7}.
$$

Next, rotate the **whole bound result** to mark the observation position:

- First observation: $\rho^1(h_{\text{phone signal fact},0})$.
- Middle observation: $\rho^2(h_{\text{phone signal fact},1})$.
- Last observation: $\rho^3(h_{\text{phone signal fact},2})$.

Finally, weight and bundle:

$$
h_{\text{phone signal}}
=\frac{1}{\sqrt3}
\left[
\rho^1(h_{\text{phone signal fact},0})
\oplus\rho^2(h_{\text{phone signal fact},1})
\oplus\rho^3(h_{\text{phone signal fact},2})
\right].
$$

Each contribution receives weight $1/\sqrt3\approx0.577$.

Bundling is commutative, but each value was attached to its observation position **before** bundling. That is how the representation distinguishes weakening signal from improving signal.

## 9. The context component uses longer descriptions of where each measurement belongs

Here is where the second use of permutation appears.

Before attaching observation order, the encoder rotates individual role hypervectors according to their position along a dependency path.

**Yes—a path through the connected graph.** In this example:

**Phone → serving cell → backhaul link**

The phone connects wirelessly through a cell, and that cell uses a backhaul link to carry traffic onward. Following those relationships is following a graph path.

There are two distinct things happening:

1. **Follow the actual graph path:** use record IDs and observation time to identify which cell and link served this phone.
2. **Encode the path’s shape:** bind generic role hypervectors such as phone, serves, cell, uses and backhaul, with positional rotations.

The hypervector contains that **typed path description**, rather than the specific phone, cell or link IDs.

### The structural-position convention

| Path position | Role | Graph interpretation |
|---:|---|---|
| 0 | Our phone | Node role |
| 1 | Its serving relationship | Relationship role |
| 2 | Its serving cell | Node role |
| 3 | The cell’s “uses” relationship | Relationship role |
| 4 | Its backhaul link | Node role |
| 5 | The shared-dependency relationship | Encoding role for peers sharing that link |
| 6 | The peer-phone role | Role describing the peer measurements |

These positions are **structural tags**. They are not seconds, distances or measured network delays.

One subtlety: positions **0–4 alternate nodes and relationships**. Phone → cell → backhaul is a **two-edge graph path**. Position 4 does not mean four graph hops.

The peer extension describes other phones sharing the link. The source dataset supplies peer observations and their cell-to-link relationships; it does not require separate identified peer-phone or peer-subscriber records.

### Why tag structural positions?

**It makes the dependency path explicitly ordered, because binding by itself is commutative.** But “necessary” is too strong for this particular demo—we should distinguish the general reason from what the code actually needs.

Suppose we describe a relationship between a source and a destination by simply binding:

$$
h_{\text{source}}\otimes h_{\text{relationship}}\otimes h_{\text{destination}}.
$$

Because binding is commutative:

$$
h_{\text{source}}\otimes h_{\text{relationship}}\otimes h_{\text{destination}}
=h_{\text{destination}}\otimes h_{\text{relationship}}\otimes h_{\text{source}}.
$$

That product preserves **which objects and relationship are present**, but cannot distinguish their endpoint positions.

Rotating the role hypervectors according to position gives:

$$
h_{\text{source}}
\otimes\rho^1(h_{\text{relationship}})
\otimes\rho^2(h_{\text{destination}}).
$$

Swapping the endpoints now gives:

$$
h_{\text{destination}}
\otimes\rho^1(h_{\text{relationship}})
\otimes\rho^2(h_{\text{source}}).
$$

These generally produce different hypervectors. **The position tags supply a distinction that the unordered product lacks.**

In our context component, a measurement’s property role gets its object’s position too: cell load gets position 2; backhaul loss gets position 4.

That says, structurally, **“this measurement belongs here in the dependency description.”** The observation-order rotation subsequently says **“this measurement occurred first, middle or last.”**

**However, this demo already has distinct role names and fixed path shapes.** “Cell,” “backhaul,” “cell load” and “link loss” are already different roles. The graph joins also select the appropriate measurements before encoding.

So structural rotation is a reasonable way to represent ordered paths, but **we have not demonstrated that these rotations are indispensable for the current feature set**. They could be redundant here.

The existing “without paths” comparison cannot answer that question: it changes the selected network measurements and removes path roles together. To isolate the value of structural tagging, we would need to keep the joins, roles and observation-order rotations identical, and remove **only the structural shifts**. That experiment has not been performed here.

## 10. Build the context path hypervectors

Define three reusable path hypervectors:

$$
h_{\text{cell path}}
=h_{\text{phone}}
\otimes\rho^1(h_{\text{serves}})
\otimes\rho^2(h_{\text{cell}}),
$$

$$
h_{\text{link path}}
=h_{\text{cell path}}
\otimes\rho^3(h_{\text{uses}})
\otimes\rho^4(h_{\text{backhaul}}),
$$

$$
h_{\text{peer path}}
=h_{\text{link path}}
\otimes\rho^5(h_{\text{shared dependency}})
\otimes\rho^6(h_{\text{peer phone}}).
$$

They describe progressively farther context: **the serving cell**, **its upstream link**, then **peers sharing that link**.

Binding multiplies coordinates commutatively. Rotating roles according to their structural positions supplies information that an unordered product of role names would lose.

## 11. Now attach each of the five context measurements

The property role gets the position of the object it describes:

| Measurement | Path and property-role hypervector before binding the value |
|---|---|
| Cell load | $h_{\text{cell path}}\otimes\rho^2(h_{\text{load property}})$ |
| Link packet loss | $h_{\text{link path}}\otimes\rho^4(h_{\text{link loss property}})$ |
| Link delay | $h_{\text{link path}}\otimes\rho^4(h_{\text{link delay property}})$ |
| Peer mean signal strength | $h_{\text{peer path}}\otimes\rho^6(h_{\text{peer radio property}})$ |
| Peer mean packet loss | $h_{\text{peer path}}\otimes\rho^6(h_{\text{peer loss property}})$ |

Each row describes the role side of a **role-value pair**. Bind it to the selected value-bucket hypervector and the context channel marker. Then rotate the **complete term** by 1, 2 or 3 for its observation time.

For example, our middle backhaul-loss measurement is 0.9478%. Its range is 0–10%, so it selects bucket 3:

$$
\mathrm{round}\left(31\times\frac{0.9478}{10}\right)=3.
$$

Its complete weighted contribution is:

$$
h_{\text{weighted middle link loss}}
=\frac{2}{\sqrt{15}}\,
\rho^2\left(
h_{\text{context channel}}
\otimes h_{\text{link path}}
\otimes\rho^4(h_{\text{link loss property}})
\otimes h_{\text{level},3}
\right).
$$

Read inward to outward:

> “Link packet loss, on the backhaul used by our phone’s serving cell, at level 3; placed in the context component; tagged as the middle observation; weighted.”

Repeat for all five measurements at all three times. Bundle those **15 weighted terms** to obtain $h_{\text{context}}$.

For a compact expression, let $\mathcal F_{\text{context}}$ be the five selected context properties. Let $h_{\text{path and property},f}$ denote the path-and-property hypervector in the table above, and let $x_{p,f}$ be the scalar measurement at observation position $p$ for property $f$. The unweighted, observation-tagged fact is:

$$
h_{\text{context fact},p,f}
=\rho^{p+1}\left(
h_{\text{context channel}}
\otimes h_{\text{path and property},f}
\otimes h_{\text{level},q_f(x_{p,f})}
\right).
$$

With the default context weight of 2:

$$
h_{\text{context}}
=\frac{2}{\sqrt{15}}
\bigoplus_{p=0}^{2}
\bigoplus_{f\in\mathcal F_{\text{context}}}
h_{\text{context fact},p,f}.
$$

Each context term receives weight $2/\sqrt{15}\approx0.516$.

The context component has multiplier 2, but each individual context term has a slightly smaller coefficient than a phone signal term because there are more context terms.

The square-root divisors account for term counts. They do **not** guarantee equal component norms. For independent zero-mean bipolar terms, bundle magnitude typically grows with the square root of the number of terms; these divisors compensate for that growth. Actual terms can be correlated, so this is approximate balancing.

The default weights—1 for phone signal, 2 for connected context and 0.25 for handset model—are modelling choices. They do not reserve fixed percentages of final similarity for those components.

## 12. Keep the two uses of permutation distinct

They use the same rotation operator, in two places:

- **Inside the term:** rotate individual role hypervectors to tag their positions along the dependency path.
- **Outside the term:** rotate the completed measurement to tag its position in the observation sequence.

The path tags stay fixed across the three observations. The outer rotation changes.

Also, the encoder represents **ordinal position**—first, middle, last. It does not separately encode the 20-second gaps.

## 13. The handset component is much simpler

Our phone model is `model_0`. This is a category, so it gets a deterministic category hypervector rather than a numeric level:

$$
h_{\text{handset}}
=0.25\,
h_{\text{handset channel}}
\otimes h_{\text{handset property}}
\otimes h_{\text{model}\_0}.
$$

Here the role-value pair is the handset-model property role bound to the `model_0` value hypervector. The channel marker distinguishes this contribution from the numeric measurements.

There is one contribution. It has no dependency-path extension and no observation-order rotation.

The phone model is constant for this episode, so the encoder includes it once.

## 14. Finally, combine the three components

$$
h_{\text{episode}}
=h_{\text{phone signal}}
\oplus h_{\text{context}}
\oplus h_{\text{handset}}.
$$

The three component hypervectors above already include their weights; do not apply the weights again when bundling them.

All three occupy the **same 4,096 coordinates**. They are superimposed through bundling; they are not concatenated into separate sections.

The raw bundle remains float32. For comparison and search storage, it is subsequently normalized:

$$
\widehat h_{\text{episode}}
=\frac{h_{\text{episode}}}{\lVert h_{\text{episode}}\rVert_2}.
$$

The norm is a scalar measure of the complete hypervector’s Euclidean length. Dividing by it preserves direction while making that length one.

Normalization happens after the complete bundle. In the code, `Encoder.encode()` returns the raw bundle; vector storage performs normalization. Normalizing each component separately would change the relative weights.

For two represented episodes, cosine similarity can then be computed from their unit-length hypervectors:

$$
\mathrm{similarity}(\text{query},\text{candidate})
=\widehat h_{\text{query}}^{\mathsf T}\widehat h_{\text{candidate}}.
$$

This is a scalar comparison score, not another hypervector or a causal explanation.

## 15. One last distinction explains what “connected” means here

The graph determines **which measurements enter**. The encoder describes those measurements using **typed roles**.

It does not bind the identities of the subscriber, phone, cell or link. The peer-phone role describes the peer averages; the dataset does not require individual peer subscriber records.

Consequently:

- Different subscribers can produce similar representations.
- Reconnecting a cell can change the selected link measurements and peer averages.
- If all selected measurements, categories and ordering remain identical, changing identifiers alone does not change the hypervector.

Record IDs remain alongside the representation for tracing its source evidence. Labels, disruption flags and geographic coordinates are excluded from the arithmetic.

The complete construction is therefore:

**Select connected facts → bucket values → bind role-value pairs and connected meanings → tag observation order → weight → bundle → normalize.**

## Representation recap

- **Three phone signal terms**, **fifteen context terms** and one handset term form three component hypervectors, then one episode hypervector. Term counts are not counts of network objects.
- Every hypervector uses $h$ with a descriptive suffix. Binding uses $\otimes$, bundling uses $\oplus$ and permutation uses $\rho$.
- Value bucketing, range bounds, feature selection, weights and structural positions are modelling choices. The 32 levels, 4,096 coordinates and chosen bounds are not universal HDC or telecom requirements.
- Following a particular graph path selects the evidence; representing its typed shape encodes its meaning. Structural position 4 is not four graph hops, and the current comparison does not isolate the benefit of structural rotations.
- The prototype learner reuses these represented episodes rather than introducing a new feature encoder.

The implementation anchors are [the encoder](../src/encoding.py), [the connected dataset builder](../src/data.py), [the frozen configuration](../src/config.py) and [vector storage](../src/storage.py). The broader study is documented in [the results summary](../runs/reports/summary.md).

## 16. Learning: use the same primitives to build class prototypes

We now have one complete hypervector for each episode. The learner uses that same representation; it does not go back to the source records and invent a different encoding.

**There is supervised learning, but no encoder retraining or optimization loop for the HDC memory.** A reviewed incident supplies a label, and its normalized episode hypervector is bundled into the memory for that label.

### What are we learning to predict?

A **class** is a named pattern we want to recognize. The demo uses four classes:

| Class | What the observations show |
|---|---|
| Deteriorating radio | The phone’s received signal becomes weaker across the episode |
| Transient recovery | The phone’s received signal improves by the end of the episode |
| Shared transport impairment | The shared backhaul link and peer phones show packet loss; the phone’s own signal profile can vary |
| Normal service | Neither a large signal change nor the shared backhaul impairment occurs |

“Transport” here means carrying data over the network connections, not the commuter’s mode of travel.

The review label tells the learner which class memory should receive the episode. In this demonstration, an independent rule checker reconstructs the labels from the synthetic source evidence; no human reviews were collected. The labels describe observed patterns, not confirmed causes of a dropped call.

### How binding, permutation and bundling carry through to learning

| Primitive | In one represented episode | In prototype learning |
|---|---|---|
| Binding, $\otimes$ | Forms role-value pairs and attaches the phone, graph-path roles and channel meanings | Those bindings are already present in each episode hypervector and are retained when episodes are bundled |
| Permutation, $\rho$ | Tags structural path positions and first/middle/last observations | Those tags are already present, so weakening and recovery can contribute differently to class memory |
| Bundling, $\oplus$ | Combines weighted measurement contributions into one episode hypervector | Combines normalized reviewed episode hypervectors into one accumulator per class |

**The prototype update directly uses bundling.** Binding and permutation are reused through the already encoded episodes; the learner does not apply new property bindings or rotations during an update.

The class label chooses an accumulator. This implementation does not bind a label hypervector to the episode. It also does not rotate each episode according to when its review arrives. Observation order inside an episode and feedback arrival time are different things.

### Build one prototype, step by step

Suppose two reviewed incidents belong to the deteriorating-radio class. Their raw episode hypervectors are $h_{\text{episode},1}$ and $h_{\text{episode},2}$.

First normalize each complete episode:

$$
\widehat h_{\text{episode},1}
=\frac{h_{\text{episode},1}}{\lVert h_{\text{episode},1}\rVert_2},
\qquad
\widehat h_{\text{episode},2}
=\frac{h_{\text{episode},2}}{\lVert h_{\text{episode},2}\rVert_2}.
$$

Each reviewed episode contributes a unit-length hypervector. An episode with a larger raw bundle norm therefore does not automatically receive more weight in the class memory. Its internal phone signal, context and handset weighting is still represented in its direction.

Next bundle those reviewed episodes:

$$
h_{\text{deteriorating accumulator}}
=\widehat h_{\text{episode},1}
\oplus\widehat h_{\text{episode},2}.
$$

This is a second level of bundling. Earlier, we bundled measurements into an episode. Now, we bundle episodes into class memory.

Shared directions can reinforce one another across examples, while varying contributions can partially cancel. That is the intuition behind a class summary; finite-dimensional interference and variation between examples mean it is not guaranteed to retain only class-relevant information.

The **prototype** used for comparison is the normalized class accumulator:

$$
h_{\text{deteriorating prototype}}
=\frac{h_{\text{deteriorating accumulator}}}
{\lVert h_{\text{deteriorating accumulator}}\rVert_2}.
$$

The code retains the raw float32 accumulator, and normalizes it when scoring. It does not overwrite the accumulator with its unit-length prototype after each update; that would change the weighting of earlier reviews.

For a general class label $y$, with reviewed episode indices $\mathcal I_y$:

$$
h_{\text{class accumulator},y}
=\bigoplus_{i\in\mathcal I_y}\widehat h_{\text{episode},i},
$$

$$
h_{\text{class prototype},y}
=\frac{h_{\text{class accumulator},y}}
{\lVert h_{\text{class accumulator},y}\rVert_2}.
$$

Each class has its own 4,096-dimensional accumulator and a scalar review count. The accumulated hypervectors are still in the same representation space as the individual episodes.

### Incorporate another reviewed incident

When a new reviewed episode arrives with label $y$:

$$
h_{\text{class accumulator},y}
\leftarrow
h_{\text{class accumulator},y}
\oplus\widehat h_{\text{new episode}}.
$$

That is the learning update. It changes the class memory while the property roles, numeric level codebook, path convention and episode encoder stay fixed.

**Learning is a change to the accumulated class representation.** It does not require changing how signal strength, packet loss or event order are encoded.

### Predict the class of a new episode

Encode the new episode using the same frozen encoder, and normalize its complete hypervector. Compare it with each available class prototype:

$$
\mathrm{score}_y(\text{query})
=\widehat h_{\text{query}}^{\mathsf T}h_{\text{class prototype},y}.
$$

Prediction chooses the available class with the highest cosine similarity:

$$
\widehat y
=\arg\max_{y\in\mathcal Y_{\text{reviewed}}}
\mathrm{score}_y(\text{query}).
$$

Here $\mathcal Y_{\text{reviewed}}$ contains only classes with at least one review. The score is a scalar cosine similarity, not a calibrated probability.

A class without reviews cannot yet be predicted. Before any reviews, the system reports insufficient labelled memory rather than guessing from four empty accumulators.

### What changes when feedback arrives later?

Review labels become available 300 seconds after the complete episode; they stand in for analyst feedback in the demonstration. No human reviews were collected.

At a decision, the model uses only reviews that have already become available. The current episode’s future review cannot update the memory used to predict that same episode. When the label later becomes available, its normalized episode hypervector is bundled into the appropriate class accumulator for subsequent predictions.

The five-minute delay is a modelling choice, not a measured analyst turnaround time. Validation and final-test labels do not update the memory in the replay.

Updates are not guaranteed to help. A new review can move a prototype toward some cases and away from others. The demo retains review provenance and before/after accumulator hashes. Exact reversal restores the previous float32 snapshot rather than relying on rounded subtraction.

### Where the conventional ML comparison fits

The study also compares this prototype learner with **logistic regression (LR)** and a **multilayer perceptron (MLP)**, a small neural network with one hidden layer.

The learners receive the same reviewed episodes and connected measurements, but their input representations differ:

| Learner | Input representation | What learning changes |
|---|---|---|
| HDC prototype learner | The 4,096-dimensional episode hypervector built with binding, permutation and bundling | Class accumulators through bundling of reviewed episodes |
| Logistic regression | Six continuous range-scaled measurements at each of three times, plus three handset-category indicators: 21 inputs | Fitted coefficients defining class decision boundaries |
| Small MLP | The same 21 continuous inputs as LR | Fitted neural-network weights |

LR and MLP use the same numeric range scaling, but keep the continuous fractions rather than bucketing them into the HDC codebook’s 32 levels. Their input ordering preserves first/middle/last positions without HDC permutations. The three handset indicators record which category is present, with coefficient 0.25 for the selected category and zero for the others.

Both conventional models are trained with an optimization procedure. Their settings are selected using separate validation worlds and then frozen before final testing. They are not limited to one training pass.

Validation can also select additional scaling fitted only on the reviewed training examples. This does not change which source measurements the models receive.

The review budgets are 1, 2, 5, 10 and 20 examples per class. With four classes, five reviews per class means twenty reviewed training episodes. Each learner receives the same selected reviews at a given budget; separate labelled validation examples support model selection.

These are comparisons on controlled synthetic patterns. A composable representation and a simple class-memory update do not by themselves establish a universal accuracy, compute or storage advantage over conventional ML.

### How the primitives connect representation and learning

**Binding attaches values to meanings. Permutation preserves positions. Bundling first creates an episode, then accumulates reviewed episodes into class prototypes. Similarity compares a new episode with those prototypes.**

The implementation anchors for this final section are [the prototype learner and delayed-feedback replay](../src/learning.py) and [the LR/MLP comparison](../src/classifiers.py).

## 17. Potential enhancements: more geographic capabilities

**This demo only scratches the surface of GeoArrow and GeoDataFusion.** GeoArrow carries observation points with coordinate-system metadata. Lance applies the geographic and time filters, and GeoDataFusion independently checks that they select the same observations. LanceDB then ranks eligible episode hypervectors by cosine distance. These are distinct operations: geographic selection determines which incidents qualify; hypervector similarity compares their represented telecom patterns.

**Lance's type system is Arrow-native.** It uses Apache Arrow types and in-memory arrays, with support for extension-type metadata. This shared foundation lets us build on compatible developments across the larger Arrow ecosystem, including GeoArrow geometry types and GeoDataFusion spatial queries. The demo already checks that GeoArrow geometry and coordinate-system metadata survive the Lance round trip. Additional operations still need integration and validation in the relevant query engine; shared types do not automatically make every operation available inside Lance. [Lance data types](https://lance.org/guide/data_types/), [Lance schema and extension types](https://lance.org/format/table/schema/)

Potential extensions include:

- **Richer geographic data with GeoArrow.** Carry road lines, service-area polygons and infrastructure locations alongside observation points, and exchange those geometries with GeoPandas or GeoParquet while preserving geometry and coordinate-system metadata. [GeoArrow documentation](https://github.com/geoarrow/geoarrow-python)
- **Spatial joins with GeoDataFusion.** Match observations to service-area polygons using containment or intersection, then count or summarize incidents by area. These queries would extend its current role as an independent filter check. [GeoDataFusion spatial relationships](https://github.com/datafusion-contrib/geodatafusion#spatial-relationships)
- **Distance-based candidate selection.** Use geometry distance to select observations near a road or infrastructure location, or to support a geographic-radius filter. GeoDataFusion supports `ST_Distance`; a metre-based query would first require geometries in an appropriate coordinate system whose units are metres. Our current longitude/latitude geometry calculations use angular units. [GeoDataFusion measurement functions](https://github.com/datafusion-contrib/geodatafusion#measurement-functions)

For example, a future query could **find incidents within 500 metres of a location, apply the time and earlier-memory filters, then rank their hypervectors by similarity in LanceDB**. That would extend candidate selection while reusing the existing encoder and prototype learner. Geography could remain outside the hypervector; adding geographic features to the encoding would be a separate modelling choice.
