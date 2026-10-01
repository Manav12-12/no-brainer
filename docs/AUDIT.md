# Engineering audit trail

## Jev reflex gate

The original reflex gate assumed that the `known_pattern` `noul` score was a
calibrated probability with a natural cutoff of 0.5. Jev 1.13.0 did not use
that range on the cached UNSW sample: all 400 scores were between 0.20 and
0.38, and the combined reflex confidence never exceeded 0.60. Consequently,
the former `known >= 0.5` and `confidence >= 0.8` gate could not fire.

No independent cached Jev validation run exists, and a new paid/network call
was not authorized. `scripts/tune_jev_gate.py` therefore makes a deterministic,
attack-family-stratified 50/50 partition of the existing cache. It tunes only
on the validation partition, maximizing detection subject to a 5% validation
false-action cap, and evaluates once on the other partition. This is weaker
than a genuinely independent validation run because Phase 1 already reported
aggregate statistics for all 400 records; the limitation is explicit rather
than hidden.

The selected gate is:

- `known_pattern >= 0.27`
- combined reflex confidence `>= 0.28`
- Jev must select a non-`no_op` closed-set action

Operational values live in `configs/jev.yaml`. The known-pattern boundary is
an empirical Jev 1.13.0 operating point, not a universal interpretation of a
`noul` score and not a claim that Jev is well calibrated.

## Synthetic-brain propagation

The original 48-node proxy gave each PN one ORN input with connectivity 4,
equivalent to 1.1 mV under the published 0.275 mV/contact model constant. No
PN spiked in held-out traffic or in maximal-input probes as long as 1 second.

The local FlyWire v783 connectivity asset was measured before changing the
proxy. Across excitatory edges, the 99th percentile is 31 contacts; median
excitatory in-degree is 37 edges. The reduced graph cannot preserve that
in-degree with its former one-to-one feed-forward layers. It now uses complete
feed-forward connections and treats each proxy edge as an aggregate bundle of
31 contacts. This is a clearly labeled engineering proxy derived from the
asset distribution, not a claim about identified ORN or PN neurons in
FlyWire. The membrane constants and 0.275 mV/contact value remain unchanged
and retain their citations in `configs/brain_mb_subnet.yaml`.

`scripts/run_reachability_probe.py` records the FlyWire measurements and 100
maximal-input runs at 15, 20, 50, and 100 ms in
`artifacts/brain-reachability.json`. Interactive and evaluation paths use the
configured 50 ms duration rather than their former inconsistent 15/20 ms
hard-coded durations.

## Offline KC-to-MBON learning

The original `offline_kc_mbon_update()` was reachable only from tests. Its
returned array was never copied into graph or Brian2 weights. Evaluation was
therefore always static despite the presence of feedback-related modules.

`train_kc_mbon_weights()` now runs a bounded offline training phase and writes
each updated KC-to-MBON value back to the graph edge attribute consumed by the
next `simulate_lif()` call. The teaching signal is a reward-prediction error:
the binary training label minus normalized MBON activity. Missing edges in
shuffled/random controls remain missing; learning does not repair topology.
Weights are clipped to positive proxy-connectivity bounds `[0.1, 100]`.

Synthetic and public-data evaluations train on a deterministic balanced batch
of 32 training records. Evaluation is explicitly labeled `trained`; there is
no ambiguous implicit/static mode. The 0.05 learning rate is an engineering
choice, and no claim is made that this is a biological learning rule.

## Homeostatic drive

The escalation path now maintains persistent pressure independently for each
host. Every uncertain event contributes a composite stimulation value made
from normalized sensory-encoder rate and propagated KC novelty. A three-event
sliding window feeds a leaky accumulator. An isolation action fires at the
configured threshold, immediately reduces drive to 15% of its pre-action
value, and clears the stimulation window. Isolation changes the cyber-range
state; the attacker then emits `contained` telemetry with no attack offset, so
subsequent observations continue to reduce drive through decay rather than
depending only on the immediate reset.

`scripts/tune_drive.py` used 10 attack and 10 benign validation episodes with
seeds disjoint from evaluation. It selected window 3, decay 0.82, stimulation
floor 0.15, and threshold 0.4 under a 5% false-action cap. Validation reached
100% episode detection, 0% false actions, and mean containment step 4.7. The
selection and exact seed ranges are recorded in
`artifacts/drive-validation.json`; held-out seeds were not used.
