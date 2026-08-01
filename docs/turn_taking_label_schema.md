# Turn-Taking Label Schema

This is the labeling contract for the corpus that trains the Phase 3 turn-taking
model. It is written before any labeling happens, per the Phase 1.1 gate, so that
every annotated pause — CANDOR or self-recorded roleplay — is labeled the same way.

## 1. What gets labeled

The unit of labeling is a **candidate pause**: a contiguous stretch of non-speech
in one speaker's channel, bounded by that speaker's own speech on either side (or
by the end of the conversation). Frames *during* active speech are not decision
points — a speaker mid-utterance is trivially not done — so labeling effort goes
entirely into the pauses, which is also where the fixed-threshold VAD baseline
(Phase 3.1) makes its latency/interruption tradeoff.

Each candidate pause gets exactly one binary label:

| Label | Meaning |
|---|---|
| `1` — `TURN_COMPLETE` | The speaker is done. The floor should pass; an agent listening here should start responding. |
| `0` — `MID_TURN_PAUSE` | The speaker isn't done. They will resume the same communicative act after this pause. |

### Why per-pause, not per-frame, storage

The project spec calls for a label "per 100ms frame." Within one candidate pause
the label is constant by construction — a pause doesn't become "more complete" the
longer it lasts, it either is or isn't the end of the turn (that's what makes mid-turn
pauses like "my budget is... around..." the hard case: they can be arbitrarily long
and still be label `0`). So labels are stored **per pause** in the raw label file
(one row per pause, see §4) to keep hand-labeling tractable, and the frame-level view
Phase 3.2's feature pipeline needs is a deterministic expansion — one row per 100ms
tick between `pause_start_ms` and `pause_end_ms`, all carrying the pause's label.
That expansion is a materialization step, not a labeling decision, and belongs in
Phase 3.2's data loader, not here.

## 2. Decision rule

Label a pause `TURN_COMPLETE` (1) only if, using the **full conversation as hindsight**
(not just the audio up to the pause), the speaker does not resume the same
communicative act after this pause — the next event is a turn change, a genuinely
new/unrelated utterance, or the conversation ending. Everything else is `0`.

This is a retrospective, offline label. It is deliberately allowed to see the future
(what the speaker says next) even though the runtime model (Phase 3.3) will only ever
see the past — that gap between offline hindsight and online causality is exactly
what makes this a learning problem instead of a rule.

## 3. Hard negatives: mid-turn pause taxonomy (label 0)

Mine these deliberately and over-sample them relative to their natural frequency —
they are the examples that separate a learned endpointer from a silence timer.

| Type | Example | Why it's hard |
|---|---|---|
| Trailing-off-then-continuing | "my budget is... around 500" | Prosody and partial syntax look complete; speaker is retrieving a value |
| Filled pause / disfluency | "I want to— um— return this" | Silence-only VAD sees a long gap |
| List continuation | "I need a refund, a return label... and also—" | Sounds complete after each item |
| Self-correction / repair | "I want the red one — actually, no, the blue" | Looks done right before the correction |
| Thinking pause before specifics | "my order number is... [pause] ...4471" | Common right before numbers, dates, addresses — high-stakes for e-commerce support |
| Backchannel-eliciting pause | "so basically— [pause] —yeah?" | Speaker is checking for acknowledgment, not yielding the floor |

## 4. True positives: turn-complete taxonomy (label 1)

| Type | Example |
|---|---|
| Complete statement + silence | "I'd like to return this jacket." |
| Question awaiting answer | "Can I get a refund instead of store credit?" |
| Explicit yield cue | "...that's it, thanks." |
| Complete propositional content + falling intonation | "It arrived damaged." |

## 5. Scope and exclusions

- **Speaker channel:** for self-recorded roleplays, label the *customer* speaker's
  pauses — that's the channel a deployed endpointer listens to. The *agent* roleplay
  speaker's pauses are out of scope for this corpus. CANDOR is a natural dyadic corpus
  with no fixed customer/agent roles; label both channels there, since the goal in
  that corpus is general turn-completion signal, not domain-specific role behavior.
- **Excluded regions:** overlapping speech / crosstalk, non-speech audio (hold music,
  dead air before a session starts), and any pause shorter than 100ms (below frame
  resolution, not a meaningful decision point).
- **Ambiguous / unlabelable pauses:** mark `hard_negative_type: "ambiguous"` and
  `label: null` rather than forcing a guess. Track the rate of these — a high rate
  is itself a signal the schema needs revision.

## 6. Label file format

One JSONL row per labeled pause (see `data/turn_taking/labels/schema_example.jsonl`):

```json
{
  "conversation_id": "candor_0001",
  "pause_id": "candor_0001_p07",
  "corpus": "candor",
  "speaker": "customer",
  "pause_start_ms": 18420.0,
  "pause_end_ms": 18930.0,
  "preceding_text": "my budget is... around",
  "label": 0,
  "hard_negative_type": "trailing_off_then_continuing",
  "annotator": "varshith",
  "annotated_at": "2026-08-01T00:00:00Z"
}
```

`hard_negative_type` is required when `label == 0`, `null` when `label == 1`.
`label` may be `null` with `hard_negative_type: "ambiguous"` per §5.

## 7. Inter-annotator / test-retest reliability (the Phase 1.1 gate)

Before labeling the full corpus:

1. Label a fixed set of **100 pause samples** (mixed CANDOR + roleplay, oversampled
   for hard negatives per §3).
2. Wait **one week**, then relabel the same 100 samples blind (no access to the
   original labels) — either by the same annotator (test-retest) or a second
   annotator (inter-annotator), whichever is available.
3. Compute **Cohen's κ** between the two passes over the binary `label` field with
   `scripts/label_agreement.py` (excludes rows where either pass used `label: null`).
4. Interpret using the standard Landis & Koch bands: `<0.20` slight, `0.21–0.40` fair,
   `0.41–0.60` moderate, `0.61–0.80` substantial, `>0.80` almost perfect.

**Gate:** κ ≥ 0.60 (substantial agreement) before scaling to the full corpus. Below
that, the disagreement cases are the signal — pull them out, check which taxonomy
category they cluster in (§3/§4), and tighten the schema rather than the labels.
