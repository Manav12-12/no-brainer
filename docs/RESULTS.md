# Results

## Closed-loop result

The system now demonstrates the complete defensive cycle:

1. uncertain events ascend past the fast reflex path;
2. encoded sensory pressure and propagated KC activity stimulate per-host drive;
3. a three-event sliding window and leaky accumulator build pressure;
4. crossing 0.4 triggers simulated host isolation;
5. the attacker changes to `contained`, removing malicious telemetry offsets;
6. drive receives immediate 85% relief and continues falling on later events.

The integration test checks the state transition and post-action decline. The
live recording shows the same values streamed from Python.

## Validation-only tuning

`scripts/tune_drive.py` evaluated 10 attack and 10 benign validation episodes
using seeds 30000–30009 and 31000–31009. It selected:

| Parameter | Value |
| --- | ---: |
| Window | 3 events |
| Decay | 0.82 |
| Stimulation floor | 0.15 |
| Action threshold | 0.40 |
| Relief fraction | 0.15 |

Validation reached 100% attack-episode detection, 0% benign-episode false
actions, and mean containment step 4.7. Evaluation seeds were not used during
selection. Exact output is in `artifacts/drive-validation.json`.

## Held-out synthetic evaluation

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

The configured 32-record balanced training batch changed all 96 KC-to-MBON
edges from 31.00 to 31.96, an L1 delta of 92.16 in the measured run. Updated
weights are written into the graph consumed by later Brian2 simulations. B5
has no eligible KC-to-MBON edges and therefore records zero learning delta.

## Controls and interpretation

B4 and B5 also contained every episode. They were slower than B2/B3, but this
small synthetic evaluation is not an ablation result and does not support a
claim that FlyWire-specific topology caused the outcome. The composite drive
includes a shared sensory-pressure term; that term can carry the random graph.

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

Final checks passed Ruff, strict mypy, 58 tests with 87.84% branch-aware
coverage, Bandit, detect-secrets, and manifest verification. Default tests
continue to block network sockets; no live Jev calls were made.
