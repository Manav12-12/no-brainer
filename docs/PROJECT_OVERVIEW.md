# Project overview

## Purpose

Drosophila Sentinel is a safe, offline-first cyber-range for testing a layered
defensive controller. It combines a fast typed reflex with a small Brian2
connectome proxy and persistent per-host homeostatic drive. All hosts, attacks,
telemetry, and actions are simulated Python objects.

## Honest contribution

The project contributes a working and audited control architecture, not a new
state-of-the-art detector. It demonstrates a complete reflex-and-escalation
cycle: every event produces upstream stimulation, pressure accumulates over
time, a response changes the simulated environment, and relief follows the
action.

The strongest result is explicitly **sequential and episode-level**: on 10
attack and 10 benign synthetic validation episodes, the dual-input drive
reached 100% attack detection, 0% benign false actions, and mean containment
step 9.0. The separate **record-level** mixed benchmark has high false-positive
rates and does not support deployment. It also does not show that fly-brain
structure matters.

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

A fourth detection-quality defect was found during final validation: the pain
extractor ranked attacks below benign records. Replacing the inverted score
raised validation ROC-AUC from 0.373 to 0.641. This is the clearest detector-side
engineering improvement made by the project.

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
thresholds, combined F1 was 0.692 versus B6 at 0.712, and precision was 0.563
versus 0.569. Combined FPR was slightly lower, 0.700 versus 0.720, but both are
unusable. Most of the combined system's value comes from the independently
added B6 anomaly signal. KC-to-MBON learning added only about 0.008 validation
ROC-AUC over the untrained dual-input graph.

B4 and B5 reached higher ranking and F1 scores than the intended graph in the
final mixed benchmark, after matching it in earlier rounds. The
connectome-specific question is settled: the project found no detection benefit
from the intended fly-inspired topology. The architecture is more expressive,
but it is not a better production detector than standard anomaly detection.

The live browser receives each drive value from Python over server-sent events.
The proof clip in `docs/assets/drosophila-sentinel-drive-cycle.mp4` shows drive
crossing its threshold, isolation firing, and pressure relief.

## Locked evidence

Final metrics and threshold curves are locked in
`artifacts/detection-quality.json`, sequential validation is in
`artifacts/drive-validation.json`, and the learning comparison is in
`artifacts/kc-learning-validation.json`. No further threshold tuning,
KC-to-MBON training, or held-out evaluation is part of the final result.

## Demo guide

The live demo uses the local model and cached assets. It does not need a Jev API
key and makes no live Jev request.

### First-time setup

```bash
git clone git@github.com:Manav12-12/no-brainer.git
cd no-brainer
./scripts/fetch_offline_assets.sh --runtime-only
make setup
```

### Start the demo

```bash
make live
```

The command opens `http://127.0.0.1:8765`. If the browser does not open, visit
that address manually. Press `Ctrl+C` in the terminal to stop the server.

For a different port or a machine where automatic browser launch is unwanted:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_live_simulation.py \
  --port 8877 --no-open
```

### What to show

1. Start a new live run and point out that Python emits each event over
   server-sent events.
2. Show the reflex pain value and measured Brian2 spikes changing as events
   arrive.
3. Follow one host's `HOMEOSTATIC DRIVE` bar as pressure accumulates.
4. Show the threshold crossing, simulated defensive action, attacker transition
   to `contained`, and immediate pressure relief.
5. End with the control result: shuffled and random graphs matched or beat the
   intended topology. The demo proves the closed-loop controller, not a
   connectome-specific detection advantage.

Use `New live run` to restart. `Fullscreen` prepares a clean presentation view.
`Record WebM` starts a fresh run and records the 1920 by 1080 canvas. Use
`Stop + save` to download it.

### Troubleshooting

- If dependencies are missing, run `make setup` again.
- If port 8765 is busy, use the alternate-port command above.
- If the browser remains blank, keep the terminal open and reload the local
  page after the server prints `Live simulation ready`.
- Do not export `TYPESAFE_API_KEY` for this demo. The local path is deliberate.

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
- B4/B5 consistently match or outperform the intended topology. This rules out
  a connectome-specific benefit in the completed experiments.
- KC-to-MBON learning adds only about 0.008 validation ROC-AUC beyond the
  untrained dual-input graph; most combined value comes from B6 anomaly input.
- The Jev result covers one cached model version. Its semantically named v2
  request could not be measured without new authenticated calls.
- The mixed Jev partition is balanced and source-disjoint from B6 training but
  had previously been used for aggregate gate reporting.
- Host roles and segments in the mixed benchmark are synthetic, so it does not
  represent genuine per-host temporal context.
- High false-positive rates make every evaluated operating point unsuitable
  for deployment.
- Sequential dual-input drive tuning reached 100% validation episode detection,
  0% benign-episode false actions, and mean containment step 9.0. This synthetic
  result does not remove the mixed record benchmark's high-FPR limitation.
- Synthetic episodes do not represent operational prevalence or costs.

See `docs/AUDIT.md` for design decisions and `docs/RESULTS.md` for complete
metrics.
