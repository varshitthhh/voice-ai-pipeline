"""Phase 1.1 gate: Cohen's kappa between two labeling passes over the same
100 pause samples, taken a week apart (test-retest) or by two annotators
(inter-annotator) — see docs/turn_taking_label_schema.md Section 7.

Matches rows by `pause_id`, drops any pair where either pass left `label`
null (ambiguous, per schema Section 5), and reports kappa plus the
Landis & Koch interpretation band.

Run:
    python scripts/label_agreement.py --pass1 <path.jsonl> --pass2 <path.jsonl>
"""

import argparse
import json
from pathlib import Path

GATE_KAPPA_MIN = 0.60  # docs/turn_taking_label_schema.md Section 7


def load_labels(path: Path) -> dict:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["pause_id"]] = row["label"]
    return labels


def cohens_kappa(pairs: list) -> tuple:
    """pairs: list of (label_a, label_b) over a shared, fixed set of classes."""
    classes = sorted({v for pair in pairs for v in pair})
    n = len(pairs)

    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for a, b in pairs:
        confusion[a][b] += 1

    p_observed = sum(confusion[c][c] for c in classes) / n

    row_marginal = {c: sum(confusion[c].values()) / n for c in classes}
    col_marginal = {c: sum(confusion[r][c] for r in classes) / n for c in classes}
    p_expected = sum(row_marginal[c] * col_marginal[c] for c in classes)

    kappa = 1.0 if p_expected == 1.0 else (p_observed - p_expected) / (1 - p_expected)
    return kappa, p_observed, p_expected, confusion, classes


def interpret(kappa: float) -> str:
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass1", required=True, type=Path)
    parser.add_argument("--pass2", required=True, type=Path)
    args = parser.parse_args()

    labels1 = load_labels(args.pass1)
    labels2 = load_labels(args.pass2)

    shared_ids = sorted(set(labels1) & set(labels2))
    missing = (set(labels1) ^ set(labels2))
    if missing:
        print(f"warning: {len(missing)} pause_id(s) present in only one pass, skipped")

    pairs = []
    excluded_null = 0
    for pid in shared_ids:
        a, b = labels1[pid], labels2[pid]
        if a is None or b is None:
            excluded_null += 1
            continue
        pairs.append((a, b))

    if len(pairs) < 2:
        raise SystemExit(f"only {len(pairs)} usable label pairs — need at least 2 to compute kappa")

    kappa, p_o, p_e, confusion, classes = cohens_kappa(pairs)

    print(f"matched pause_ids: {len(shared_ids)} (excluded {excluded_null} with a null label)")
    print(f"usable pairs: {len(pairs)}")
    print("confusion matrix (rows=pass1, cols=pass2):")
    header = "        " + "  ".join(f"{c!s:>6}" for c in classes)
    print(header)
    for r in classes:
        print(f"  {r!s:>4}  " + "  ".join(f"{confusion[r][c]:>6}" for c in classes))
    print(f"observed agreement p_o = {p_o:.4f}")
    print(f"expected agreement p_e = {p_e:.4f}")
    print(f"Cohen's kappa = {kappa:.4f} ({interpret(kappa)})")

    if kappa >= GATE_KAPPA_MIN:
        print(f"PASS: kappa >= {GATE_KAPPA_MIN} gate - schema is reliable enough to scale up")
    else:
        print(f"FAIL: kappa < {GATE_KAPPA_MIN} gate - revise the schema per disagreement cases before scaling")


if __name__ == "__main__":
    main()
