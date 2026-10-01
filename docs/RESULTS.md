# Results

## Final conclusion

The project's demonstrated contribution is a working, audited, closed-loop
reflex-and-escalation architecture. It implements a real stimulus-buildup,
action, containment, and relief cycle while preserving an offline-first audit
trail. Its strongest result is the **sequential episode-level** drive
validation: 100% attack-episode detection, 0% benign-episode false actions, and
mean containment step 9.0 on 10 attack and 10 benign synthetic validation
episodes.

That result must not be conflated with the **record-level** mixed benchmark.
On mixed records, the combined system is not a better detector than standard
anomaly detection, and its false-positive rate is unsuitable for deployment.
Across every evaluation round, the B4 shuffled and B5 random controls matched
or outperformed the intended graph. The connectome-specific question is
therefore settled for this project: these experiments provide no evidence that
the fly-connectome structure improves detection.

## Closed-loop result

The system now demonstrates the complete defensive cycle:

1. every typed reflex result emits pain upstream, whether or not it acts;
2. Jev pain and independent statistical anomaly channels stimulate the brain;
3. a three-event sliding window and leaky accumulator build pressure;
4. crossing the validation-selected threshold triggers a response strategy;
5. the attacker changes to `contained`, removing malicious telemetry offsets;
6. drive receives immediate 85% relief and continues falling on later events.

The integration test checks the state transition and post-action decline. The
live recording shows the same values streamed from Python.

## Drive validation

The original pain-only `scripts/tune_drive.py` run evaluated 10 attack and 10
benign validation episodes using seeds 30000–30009 and 31000–31009. It selected:

| Parameter | Value |
| --- | ---: |
| Window | 3 events |
| Decay | 0.82 |
| Stimulation floor | 0.15 |
| Action threshold | 0.40 |
| Relief fraction | 0.15 |

That historical architecture reached 100% attack-episode detection, 0% benign
false actions, and mean containment step 4.7. After adding the independent
anomaly channel, the same predeclared search selected window 3, decay 0.75,
stimulation floor 0.25, and threshold 1.0. It reached 100% attack-episode
detection, 0% benign-episode false actions, and mean containment step 9.0.
`artifacts/drive-validation.json` records the selection. Evaluation seeds were
not used. This is sequential, episode-level synthetic validation; it is not the
mixed record-level benchmark below.

## Held-out synthetic evaluation

This table records the earlier pain-only architecture and is retained as a
historical closed-loop result. It is not the mixed detection-quality benchmark
reported below.

The final evaluation ran once on seeds 101, 202, 303, 404, 505, 606, 707, 808,
909, and 1010. Each arm received 40 steps per episode.

| Arm | Episode detection | Event detection | False-action rate | Mean TTC | P95 TTC | Actions | Injuries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 Jev replay | 0% | 0% | 0% | — | — | 0 | 0 |
| B2 trained proxy + drive | 100% | 20.00% | 0% | 5.1 | 6.0 | 10 | 0 |
| B3 Jev-to-brain + drive | 100% | 20.00% | 0% | 5.1 | 6.0 | 10 | 0 |
| B4 degree-preserving shuffle | 100% | 14.94% | 0% | 6.8 | 8.0 | 10 | 0 |
| B5 random sparse graph | 100% | 13.55% | 0% | 7.5 | 8.55 | 10 | 0 |
| B6 local baseline | 100% | 86.67% | 2.11% | 2.1 | 2.55 | 200 | 8 |

**Episode detection** means the episode reached containment. **Event
detection** is the legacy fraction of active attack events on which an action
was issued. It is lower for B2–B5 because each arm acts once and then stops the
attack. **TTC** is the simulation step where containment was first observed.

B6 is materially faster. B2/B3 are more action-sparse in this simulation, but
the evaluation does not establish that this tradeoff generalizes to real
traffic.

## Mixed detection-quality evaluation

The detection benchmark uses the pre-existing locked Jev cache partition: 100
benign and 100 malicious records spanning ten attack families. B6 training
excludes all 400 cached calibration records, giving zero source-row overlap.
Thresholds and score scaling were selected on the other 200 cached validation
records. The locked partition had previously been used for aggregate gate
reporting, so it is not a pristine never-observed test set; this is the only
mixed set for which real cached Jev responses exist without new network calls.

| Method | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev reflex | 0.534 | 0.940 | 0.681 | 0.820 | 0.651 | 0.635 |
| Brain/drive, pain only | 0.508 | 0.980 | 0.669 | 0.950 | 0.645 | 0.640 |
| Combined Jev + B6 + trained brain | 0.563 | 0.900 | 0.692 | 0.700 | 0.734 | 0.728 |
| B4 degree-preserving shuffle | 0.571 | 0.890 | 0.695 | 0.670 | 0.738 | 0.742 |
| B5 random sparse graph | 0.591 | 0.910 | 0.717 | 0.630 | 0.749 | 0.754 |
| B6 anomaly baseline | 0.569 | 0.950 | 0.712 | 0.720 | 0.725 | 0.725 |

The combined score ranks records marginally better than B6: ROC-AUC is 0.734
versus 0.725. That ranking gain does not translate into better decisions at the
selected thresholds. Combined F1 is lower (0.692 versus 0.712) and precision is
lower (0.563 versus 0.569). Its FPR is marginally lower than B6's (0.700 versus
0.720), but a 70% false-positive rate is still unusable. B4 and B5 both match or
outperform the intended topology here, consistent with every earlier control
round. The settled finding is that the combined system does not establish a
connectome-specific advantage and is not a better detector than B6.

`artifacts/detection-quality.json` contains detection-rate/FPR pairs at every
threshold from 0.0 through 1.0 in increments of 0.05.

### Jev discrimination diagnosis

The original pain calculation discarded and inverted useful ordering. Fixing
that extraction bug is a concrete engineering result:

| Validation score | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Old pain | 0.373 | 0.427 |
| `mean(action confidence, 1 − known probability)` | 0.641 | 0.656 |

The v1 state retains all 40 permitted numeric values but anonymizes their
names. Jev sees `feature_00` through `feature_39` rather than duration, byte,
TTL, load, packet, TCP, and connection-history semantics. Its role and segment
fields are synthetic cycling values, and `recent_event_count` is actually a
source-row modulus. A semantically named v2 candidate was prepared, but it has
zero cache hits and would require 200 authenticated validation calls. No API
credential was available, so no revised-prompt result is claimed.

### Independent anomaly channel and learning

Pain and anomaly are now encoded on separate ORN halves. The anomaly signal is
an IsolationForest/autoencoder ensemble calibrated without held-out labels.
KC-to-MBON training changed all 96 eligible intended-graph edges. On validation,
training improved the combined graph only from ROC-AUC 0.6592 to 0.6673 and
PR-AUC 0.6557 to 0.6597. Most useful discrimination comes from the independent
anomaly signal, not learned MBON weights. The brain component added only about
0.008 ROC-AUC over the untrained dual-input graph. B5 changed no effective
weight despite having 19 eligible edges because its observed KC update was zero.

UNSW source/destination addresses were intentionally excluded, and role/segment
metadata are synthetic. The mixed benchmark therefore measures record-level
neural pressure, not genuine per-host temporal drive sequences.

## Root-cause fixes

### Jev gate

The former `known >= 0.5` and confidence `>= 0.8` gate fired on none of 400
cached records. A family-stratified cached validation split selected
`known >= 0.27` and confidence `>= 0.28`. It achieved 9% detection and 5% false
actions on validation; the locked partition achieved 7% and 0%. This remains a
weak reflex and is not presented as good calibration.

### Signal propagation

Before the fix, maximal 150 Hz stimulation produced no PN or deeper spikes,
including one-second probes. The reduced graph now uses convergent feed-forward
connectivity and proxy edges representing 31 contacts, the measured P99
excitatory edge connectivity in the local FlyWire v783 asset.

| Duration | PN active runs | KC | MBON | Descending |
| --- | ---: | ---: | ---: | ---: |
| 15 ms | 77/100 | 2/100 | 0/100 | 0/100 |
| 20 ms | 100/100 | 81/100 | 34/100 | 0/100 |
| 50 ms | 100/100 | 100/100 | 100/100 | 100/100 |
| 100 ms | 100/100 | 100/100 | 100/100 | 100/100 |

This is an engineering proxy, not a reconstruction of biological olfactory
pathways. Exact measurements are in `artifacts/brain-reachability.json`.

### Learning connection

The current dual-input 32-record batch changes all 96 eligible intended-graph
edges and writes them into the graph consumed by later Brian2 simulations.
The mixed validation result above shows that this learning has only a small
effect beyond the untrained graph.

## Controls and interpretation

B4 and B5 also contained every episode. Across the historical synthetic run,
validation analysis, and final mixed benchmark, these controls consistently
matched or outperformed the intended topology. This is now a settled negative
result: the tested FlyWire-inspired structure contributes no measurable
detection advantage. The composite drive includes shared sensory and anomaly
terms that can carry shuffled and random graphs.

The held-out UNSW Worms set contains only attacks and cannot provide a false-
action rate. Historical public-data metrics should not be conflated with the
closed-loop synthetic episode metrics above.

## Visual evidence

- `docs/assets/drosophila-sentinel-drive-cycle.mp4`: 24.02-second, 1920×1080
  H.264 proof clip.
- `docs/assets/drosophila-sentinel-drive-cycle.webm`: original browser capture.
- `docs/assets/drive-action-relief.png`: frame showing threshold crossing and
  animated pressure relief.

## Verification

Final checks passed Ruff, strict mypy, 60 tests with 88.85% branch-aware
coverage, Bandit, detect-secrets, and manifest verification. Default tests
continue to block network sockets; no live Jev calls were made.
