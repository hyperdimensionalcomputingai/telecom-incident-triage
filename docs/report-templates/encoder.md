## How the encoder is built

The encoder turns an episode into one **{{ dimension }}-dimensional hypervector**. Think of it as an additive description: a measurement contributes according to what it measures, where it sits in the dependency path, and when it occurs. The result retains those distinctions while supporting a single similarity comparison.

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

The **phone signal channel** contains the commuter phone's own signal measurement. The **connected-context channel** contains the other five measurements: cell load, backhaul loss and latency, and the two peer averages, all joined through the active dependency at the same time. Three observation times produce three phone signal facts and fifteen context facts per episode. The phone's handset model is a separate, small categorical contribution.

The graph chooses which source measurements enter the representation. The encoder binds **typed roles**, such as phone → serving cell → backhaul; it does not bind subscriber, cell or link identifiers. Two unrelated subscribers can therefore resemble each other when their measurements and connected context match. Changing an edge can change the joined facts even when the full network contains the same measurements.

### 2. Make nearby numbers similar

For a measurement $x$ with physical range $[a_f,b_f]$, first map it to a clipped fraction, then to one of {{ levels }} numeric levels:

$$
s_f(x)=\mathrm{clip}\!\left(\frac{x-a_f}{b_f-a_f},0,1\right),
\qquad q_f(x)=\mathrm{round}\!\left((K-1)s_f(x)\right).
$$

TorchHD supplies a seeded family of correlated level hypervectors $\ell_0,\ldots,\ell_{K-1}$. Adjacent levels overlap more than distant levels. For example, −90 and −92 dBm receive nearby representations, while −90 and −115 receive more distinct ones. The ranges are fixed physical bounds, not statistics fitted to validation or test data. Values outside them are clipped; rounding introduces finite numeric resolution.

The same level family serves every numeric field. Binding each value to its attribute role distinguishes radio strength from packet loss, even if their scaled fractions coincide.

### 3. Bind the value to its meaning and connected role

We use $\otimes$ for **binding**, $\oplus$ for **bundling**, and $\rho$ for **permutation**. Here binding multiplies corresponding coordinates, bundling adds corresponding coordinates without thresholding, and $\rho^j(v)$ rotates the coordinates of $v$ by $j$ positions. The repeated-bundling symbol $\bigoplus$ combines several hypervectors. Atomic roles and channel markers are seeded bipolar arrays containing +1 and −1.

For the commuter phone's signal strength, the role product is:

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

Bundling adds these contributions. Define the phone signal channel $S$ and context channel $C$ as:

$$
S=\frac{1}{\sqrt{3}}\bigoplus_{p=0}^{2}e_{p,\mathrm{radio}},
\qquad
C=\frac{1}{\sqrt{15}}\bigoplus_{p=0}^{2}\bigoplus_{f\in\mathcal F_C}e_{p,f}.
$$

The square-root divisors account for the different numbers of terms; they do not force every episode's channel norm to be equal. If $H$ is the handset-category hypervector bound to its attribute and channel roles, the raw episode accumulator is:

$$
z=(w_S S)\oplus(w_C C)\oplus(w_H H),
\qquad (w_S,w_C,w_H)=({{ phone_signal_weight }},{{ context_weight }},{{ handset_weight }}).
$$

**“Context weight” means $w_C$, the multiplier of the bundled connected-context channel $C$ before normalization.** Context here consists of the five graph-joined network and peer measurements in the table, at each of three times; it does not mean location, subscriber identity or free text. With the active defaults, the complete equation is:

$$
z=\frac{1}{\sqrt{3}}\bigoplus_{p=0}^{2}e_{p,\mathrm{radio}}
 \oplus\frac{2}{\sqrt{15}}\bigoplus_{p=0}^{2}\bigoplus_{f\in\mathcal F_C}e_{p,f}
 \oplus0.25H.
$$

Each radio fact therefore has raw coefficient $1/\sqrt{3}\approx0.577$, while each network-context fact has $2/\sqrt{15}\approx0.516$. The channel multiplier is two, but each context fact does not receive twice the coefficient of a radio fact: the channel contains more terms. The experiment configuration and stored term manifests record these coefficients explicitly.

**Why give connected context more weight?** A shared transport incident can accompany weak, recovering or normal phone radio. Its common evidence lies upstream. Weight {{ context_weight }} was chosen in the earlier validation diagnosis and frozen before this fresh evaluation. It keeps the network evidence from being overwhelmed by the varying radio profile. It is a modelling choice for this study, not a universal HDC constant.

These are weights on raw contributions. Doubling a channel multiplies its direct contribution to a pairwise dot product by four before normalization; cross terms and normalization also affect the final score. It does not reserve a fixed percentage of similarity for that channel.

### 5. Use the same accumulator for retrieval, editing and learning

All arithmetic above uses float32 and retains the unthresholded bundle. Only then normalize:

$$
\hat z=\frac{z}{\lVert z\rVert_2},
\qquad \mathrm{similarity}(q,x)=\hat z_q^\mathsf{T}\hat z_x.
$$

Retrieval first applies exact spatial/time eligibility, then ranks eligible earlier episodes by similarity. Search storage uses float16; computation returns to float32 and the stored-vector residual is checked separately.

Because the raw bundle is retained, removing the handset means $z'=z\oplus(-w_HH)$, followed by normalization. Subtracting a component from an already normalized vector would be a different operation. The acceptance checks compare this edit with a complete rebuild.

A reviewed incident labelled $y$ updates a class accumulator by addition:

$$
A_y\leftarrow A_y\oplus\hat z,
\qquad \mathrm{score}_y(q)=\hat z_q^\mathsf{T}\frac{A_y}{\lVert A_y\rVert_2}.
$$

This is an **additive cosine class-memory classifier**: one accumulator per class, containing the sum of normalized hypervectors from that class's reviewed incidents. Prediction chooses the available class whose normalized accumulator has the highest cosine similarity to the query. The class representative is sometimes called a prototype; it is specifically this accumulated vector, not a separate feature model or neural network. The encoder stays fixed while labelled memory grows. Unseen classes are excluded from prediction; before any reviews, the system reports insufficient labelled memory. Updates retain an audit record and support exact reversal.

For an inspected candidate with weighted terms $t_j$, the retained manifest also permits exact arithmetic attribution:

$$
z_x=\bigoplus_{j=1}^{m} t_j,
\qquad a_j=\frac{\hat z_q^\mathsf{T}t_j}{\lVert z_x\rVert_2},
\qquad \mathrm{similarity}(q,x)=a_1+\cdots+a_m.
$$

The $t_j$ are hypervector contributions, combined by bundling; each $a_j$ is a scalar contribution to the cosine score. Each contribution links back to source observations, telemetry and valid edges. These contributions explain how the numeric score was assembled, including interference between terms. They do not establish the cause of a dropped call: the source records provide provenance, while similarity proposes comparisons.
