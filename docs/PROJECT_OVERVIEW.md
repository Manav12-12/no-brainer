# Project overview

## Purpose

Drosophila Sentinel is a safe, offline-first cyber-range for testing a layered
defensive controller. It combines a fast typed reflex with a small Brian2
connectome proxy and persistent per-host homeostatic drive. All hosts, attacks,
telemetry, and actions are simulated Python objects.

## Decision architecture

- **Jev reflex:** handles known, confident cases using a closed action set.
- **Brain escalation:** processes uncertain events through ORN, PN, KC, MBON,
  APL, and descending proxy populations.
- **Homeostatic drive:** accumulates sustained uncertain stimulation per host,
  decays slowly, and triggers isolation at a validated threshold.
- **Cyber-range feedback:** isolation changes attacker state to `contained`,
  which removes attack telemetry and lets drive continue to fall.

The drive input is the mean of normalized sensory pressure and propagated KC
novelty. This gives the loop a real neural signal while retaining sensitivity
to anomalous encoded telemetry.

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

The live browser receives each drive value from Python over server-sent events.
The proof clip in `docs/assets/drosophila-sentinel-drive-cycle.mp4` shows drive
crossing its threshold, isolation firing, and pressure relief.

## Reproduce

```bash
make setup
PYTHONPATH=src .venv/bin/python scripts/run_reachability_probe.py
PYTHONPATH=src .venv/bin/python scripts/tune_drive.py
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
- B5's result relies mainly on shared sensory pressure and has no KC-to-MBON
  learning edges.
- The Jev result covers one cached model version and has low detection.
- Synthetic episodes do not represent operational prevalence or costs.

See `docs/AUDIT.md` for design decisions and `docs/RESULTS.md` for complete
metrics.
