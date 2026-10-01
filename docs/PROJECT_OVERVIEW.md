# Project overview

## Purpose

Drosophila Sentinel is a safe, offline-first cyber-range for testing a layered
defensive controller. It combines a fast typed reflex with a small Brian2
connectome proxy and persistent per-host homeostatic drive. All hosts, attacks,
telemetry, and actions are simulated Python objects.

## Decision architecture

- **Jev reflex:** may act immediately and always reports a scalar pain signal.
- **Independent anomaly nociceptor:** combines IsolationForest and autoencoder
  scores without replacing Jev pain.
- **Brain:** processes pain and anomaly on separate ORN channels through PN,
  KC, MBON, APL, and descending proxy populations.
- **Homeostatic drive:** accumulates sustained neural stimulation per host,
  decays slowly, and can trigger a response strategy at a validation-selected
  threshold.
- **Cyber-range feedback:** isolation changes attacker state to `contained`,
  which removes attack telemetry and lets drive continue to fall.

The immediate reflex and upstream report are independent. A confident Jev
action does not bypass the brain. Drive combines nociceptor input, propagated
KC activity, and a trainable MBON readout. Threshold crossings select a response
strategy that can contain one host, escalate a narrow reflex, or coordinate
multiple hosts in one segment.

## What changed

Three defects blocked the original design:

1. Jev outputs never reached hard-coded probability/confidence gates.
2. ORN input could not cause a PN spike, so every deeper readout was zero.
3. offline KC-to-MBON updates were never written into evaluation weights.

The gates are now configuration-backed and validation-derived. The reduced
graph uses an explicitly documented aggregate-contact proxy based on local
FlyWire v783 statistics. Offline training mutates the same graph evaluated in
later runs. A tested drive controller closes the action-feedback loop.

## Evidence

On 10 held-out synthetic seeds, B2 and B3 contained 100% of episodes with no
false actions and mean containment time 5.1 steps. B4 and B5 also reached 100%,
at 6.8 and 7.5 steps. B6 reached 100% at 2.1 steps with a 2.11% false-action
rate. These data show a functioning controller, not a FlyWire-topology benefit.

On the mixed 200-record cached benchmark, the combined detector reached
ROC-AUC 0.734 and PR-AUC 0.728 versus B6 at 0.725/0.725. At validation-selected
thresholds, combined F1 was 0.692 versus B6 at 0.712. B4 and B5 reached higher
ranking and F1 scores than the intended graph. The architecture is more
expressive, but it is not a more accurate production detector.

The live browser receives each drive value from Python over server-sent events.
The proof clip in `docs/assets/drosophila-sentinel-drive-cycle.mp4` shows drive
crossing its threshold, isolation firing, and pressure relief.

## Reproduce

```bash
make setup
PYTHONPATH=src .venv/bin/python scripts/run_reachability_probe.py
PYTHONPATH=src .venv/bin/python scripts/tune_drive.py
PYTHONPATH=src .venv/bin/python scripts/run_detection_quality.py
PYTHONPATH=src .venv/bin/python scripts/run_experiments.py --mode replay
make live
```

The tuning command is for reproducing the recorded validation result, not for
retuning against evaluation seeds. `make live` binds only to `127.0.0.1` and
uses cached/offline reflex behavior.

## Security posture

- network sockets are blocked during normal tests;
- runtime egress is restricted to the pinned TypeSafe endpoint;
- secrets are read only from the environment;
- payloads omit labels, source rows, addresses, and timestamps;
- defensive actions affect only in-memory simulated hosts;
- available external assets and caches are SHA-256 verified.

## Limits

- The 48-node network and cyber-to-neuron encoder are engineering proxies.
- The aggregate edge parameter does not identify actual FlyWire ORN/PN paths.
- B4/B5 perform comparably enough to rule out a topology-specific claim.
- KC-to-MBON learning adds little beyond the untrained dual-input graph.
- The Jev result covers one cached model version. Its semantically named v2
  request could not be measured without new authenticated calls.
- The mixed Jev partition is balanced and source-disjoint from B6 training but
  had previously been used for aggregate gate reporting.
- High false-positive rates make every evaluated operating point unsuitable
  for deployment.
- Sequential dual-input drive tuning reached 100% validation episode detection,
  0% benign-episode false actions, and mean containment step 9.0. This synthetic
  result does not remove the mixed record benchmark's high-FPR limitation.
- Synthetic episodes do not represent operational prevalence or costs.

See `docs/AUDIT.md` for design decisions and `docs/RESULTS.md` for complete
metrics.
