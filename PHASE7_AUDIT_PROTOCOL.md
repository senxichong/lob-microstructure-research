# Phase 7 adversarial audit preregistration

MSFT 2003-01-03 only. This audit attempts to falsify the frozen Phase 1–6
pipeline; it is a separately coded audit, not an external human certification.
No new data, predictive optimization, threshold selection, model refits for
performance, or passive execution. Primary horizon remains one second.

Before new sensitivity estimates: fix 1,000 paired circular clock-block draws
at 30/60/300 seconds. At 60 seconds reproduce seed 505060 and the original
11,700-second clock with holes; at 30/300 use seeds 707030/707300. Re-rank
literal resampled observations using SciPy; never reuse the production weighted
rank implementation. Reproduce Phase 3's 1,000 60s draws with seed 314219,
1,000 large target rotations with seed 271828 and minimum shift 600 seconds,
all ten primary hypotheses for BH/BY (seven directional features alone are not
the declared testing family), and tie-preserving quintiles.

Raw replay parses the complete existing ZIP independently, initializes orders
from the start of the file and checks all retained MSFT states. Independently
derive current/rolling features and labels from Phase 1 books/events, including
multiple source ties and deterministic event windows. Audit all grid rows.
Future mutation seed 707008: five interior cutoffs; change/remove/reorder future
payloads, preserving the tested past. Settled labels stop before the cutoff.
Same-timestamp corruption must visibly change prefix features or be rejected;
no corrupted artifact becomes research input.

Model isolation uses saved fitted artifacts and a small synthetic adversarial
fit only where needed to exercise the actual fitting interface, with no real
parameter retuning. Future-derived feature trap must fail before fitting.
Audit all fitted feature lists, training preprocessing, validation selection,
fold/target timestamps, and test-label mutation. Preserve frozen artifacts.

Execution replay is a new integer-price implementation reading only saved scores,
validation thresholds and Phase 2 quotes, with source-order last-as-of selection,
one share maximum inventory, displayed-depth checks, two-share flips and the
existing test_end-1ms risk close. Independently reproduce the required model,
latency, threshold, fee, random/inversion/delay and 500 rotation results. Bootstrap
additive marked-equity increments on the original full clock, not raw transactions
spliced into fictitious inventory paths; reproduce seed 606060 and all 1,000 draws.
Assess financing separately from inventory bounds and fixed $53.60 normalization.

Document evidence per check; numerical tolerance 1e-12 for correlations/model
predictions and 1e-8 USD for accumulated cash unless explicitly recorded otherwise.
No anomaly is silently ignored. CRITICAL/HIGH implementation defects require
archiving affected original artifacts, a narrow fix and rerunning every affected
downstream analysis. Inherent limitations remain issues; no new dataset is added.
Documentation clarifications live in QUANT_AUDIT.md/claim dispositions to preserve
historical frozen reports; rewrite historical prose only if a misleading claim
requires an explicit audited correction, with old/new hashes recorded.

PASS requires completed independent checks, full tests with zero failures and no
unresolved critical defect. Integrity/signal/economic verdicts follow evidence,
not the expected conclusion. Stop after Phase 7.
