## Future work: teaching the class memory which facts matter

**What we saw.** HDC's mistakes concentrate on one pattern: {{ hard_pattern_intro }}. They are most common when other evidence in the incident points elsewhere: {{ hard_pattern_finding }} {{ lr_comparison }}

**Why it happens.** HDC's class memory is a running sum of each pattern's reviewed incidents. It learns what a *typical* incident of each pattern looks like, not which facts tell the patterns apart: every fact keeps the fixed weight the encoder gave it, and adding examples never changes those weights. When a pattern is decided by a few facts among many that vary, those few can be outvoted. {{ why_example }} LR and the MLP instead fit a separate weight to every input from the labels, so the deciding measurement can count for far more. This is a property of the running-sum class memory, not of HDC as a whole: the encoding still carries the deciding facts.

**What to try next.**

- **Mistake-driven updates.** When the memory misclassifies a reviewed incident, add it to the correct pattern and subtract it from the wrongly predicted one. This is a common extension in HDC; it keeps updates to vector arithmetic, but each update now depends on the current memory. Its effect on update cost, order independence and exact reversal needs measuring.
- **Learned weights.** Choose the weight of each fact or channel from validation data, instead of the fixed values used here.
- **Finer distinctions in the numbers.** {{ level_idea }}
- **Separate the encoding from the class memory.** Run the same running-sum classifier on the raw 21 measurements, and an error-driven learner on the hypervectors. This would show how much of the gap comes from the class memory and how much from the encoding.

<details>
<summary>Diagnostic details: where the mistakes happen and how the score splits</summary>

These numbers come from rebuilding the first seed pair's class memory (data seed {{ data_seed }}, encoder seed {{ encoder_seed }}) at {{ budget }} reviews per class, the same setting as the per-pattern table in Experiment 3. They are regenerated with the report.

**Mistakes on {{ hard_pattern }} incidents, by what the phone's own signal does.** The groups use the review rule's {{ radio_threshold }} dB threshold.

| Phone's own signal | Incidents | HDC wrong | LR wrong |
|---|---:|---:|---:|
{{ group_rows }}

**How the score splits.** For the {{ split_count }} {{ hard_pattern }} incidents where the phone's signal {{ split_group }}, each similarity score splits exactly into a phone-signal part and a network part. The table compares the correct memory with the memory HDC most often picks instead, "{{ confused_pattern }}".

| Part of the score | Toward the correct pattern | Toward "{{ confused_pattern }}" | Difference |
|---|---:|---:|---:|
{{ split_rows }}

On average the correct pattern wins by only {{ net_margin }}, so a modest pull from the phone's signal is enough to flip an incident.

{{ level_finding }}

</details>
