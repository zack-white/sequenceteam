#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_leakage.py

Leakage checks for CAFA5 Sequence Team cluster-aware splits.

What it checks:
1) Mutual exclusivity of fold ID sets (val IDs per fold).
2) MMseqs clusters do not span folds (core anti-leakage guarantee).
3) Fold size summary (helpful sanity check).

Default behavior:
- Prefer fold IDs from: splits/fold*/val_ids.txt
- Fallback to: fold0_ids.txt ... fold4_ids.txt (in project root)
- Cluster TSV default: mmseqs_out/cafa_train_si55_cluster.tsv

Exit code:
- 0 if all checks pass
- 1 if any check fails
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Set, Tuple


@dataclass
class FoldData:
    name: str
    path: str
    ids: Set[str]


def _read_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def _discover_folds(splits_dir: str, n_folds: int) -> List[FoldData]:
    """
    Try to load folds from splits/fold*/val_ids.txt first.
    If not found, fallback to fold{i}_ids.txt in current directory.
    """
    folds: List[FoldData] = []

    # Preferred: splits/fold0/val_ids.txt ...
    candidates = []
    for i in range(n_folds):
        p = os.path.join(splits_dir, f"fold{i}", "val_ids.txt")
        if os.path.exists(p):
            candidates.append((f"fold{i}", p))

    if candidates:
        for name, p in candidates:
            folds.append(FoldData(name=name, path=p, ids=_read_ids(p)))
        return folds

    # Fallback: fold0_ids.txt ... in cwd
    for i in range(n_folds):
        p = f"fold{i}_ids.txt"
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {os.path.join(splits_dir, f'fold{i}', 'val_ids.txt')} "
                f"and also missing fallback file {p} in current directory."
            )
        folds.append(FoldData(name=f"fold{i}", path=p, ids=_read_ids(p)))
    return folds


def _load_cluster_tsv(cluster_tsv: str) -> Dict[str, List[str]]:
    """
    Reads MMseqs cluster TSV with format:
      rep<TAB>member
    Returns rep -> list[members]
    """
    rep2members: Dict[str, List[str]] = defaultdict(list)
    with open(cluster_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad TSV line (expected 2 columns): {line[:200]}")
            rep, mem = parts
            rep2members[rep].append(mem)
    return rep2members


def check_fold_disjoint(folds: List[FoldData], max_examples: int = 5) -> Tuple[bool, List[str]]:
    """
    Ensure fold ID sets are mutually exclusive.
    """
    ok = True
    msgs: List[str] = []
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            inter = folds[i].ids & folds[j].ids
            if inter:
                ok = False
                ex = sorted(list(inter))[:max_examples]
                msgs.append(
                    f"[FAIL] Overlap between {folds[i].name} and {folds[j].name}: "
                    f"{len(inter)} ids (examples: {ex})"
                )
    if ok:
        msgs.append("[OK] Fold ID sets are mutually exclusive (no ID appears in multiple folds).")
    return ok, msgs


def check_cluster_spans_folds(
    folds: List[FoldData],
    rep2members: Dict[str, List[str]],
    max_bad_clusters: int = 20,
    max_member_examples: int = 10,
) -> Tuple[bool, List[str]]:
    """
    Core leakage check:
    Each MMseqs cluster's members must belong to at most one fold.
    """
    # Map protein id -> fold index
    id2fold: Dict[str, int] = {}
    for fi, fold in enumerate(folds):
        for pid in fold.ids:
            if pid in id2fold:
                # This should already be caught by disjoint check, but keep hard fail.
                return False, [f"[FAIL] ID appears in multiple folds: {pid}"]
            id2fold[pid] = fi

    bad = 0
    msgs: List[str] = []
    checked = 0
    missing_from_folds = 0

    for rep, members in rep2members.items():
        folds_seen: Set[int] = set()
        members_seen = 0
        for m in members:
            if m in id2fold:
                folds_seen.add(id2fold[m])
                members_seen += 1
            else:
                # member not present in any fold set (unexpected if folds cover all sequences)
                missing_from_folds += 1

        # If a cluster has members present in multiple folds -> leakage
        if len(folds_seen) > 1:
            bad += 1
            if bad <= max_bad_clusters:
                sample_members = members[:max_member_examples]
                msgs.append(
                    f"[FAIL] Cluster spans multiple folds: rep={rep}, folds={sorted(folds_seen)}, "
                    f"members(sample)={sample_members}"
                )

        checked += 1

    if bad == 0:
        msgs.append("[OK] No MMseqs cluster spans multiple folds (cluster-level split prevents leakage).")
    else:
        msgs.append(f"[FAIL] Found {bad} clusters spanning multiple folds (see examples above).")

    msgs.append(f"[INFO] Clusters checked: {checked}")
    if missing_from_folds > 0:
        msgs.append(
            f"[WARN] {missing_from_folds} cluster-member entries were not found in fold ID sets. "
            f"This may be OK if your folds cover only a subset, but double-check."
        )

    return bad == 0, msgs


def fold_size_summary(folds: List[FoldData]) -> List[str]:
    sizes = [len(f.ids) for f in folds]
    msgs = []
    msgs.append("[INFO] Fold sizes (val IDs):")
    for f in folds:
        msgs.append(f"  - {f.name}: {len(f.ids):,}  ({f.path})")
    if sizes:
        msgs.append(
            f"[INFO] Size stats: mean={mean(sizes):.2f}, std={pstdev(sizes):.2f}, "
            f"min={min(sizes)}, max={max(sizes)}"
        )
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="splits", help="Path to splits/ directory (default: splits)")
    ap.add_argument("--n-folds", type=int, default=5, help="Number of folds (default: 5)")
    ap.add_argument(
        "--cluster-tsv",
        default=os.path.join("mmseqs_out", "cafa_train_si55_cluster.tsv"),
        help="MMseqs cluster assignments TSV (default: mmseqs_out/cafa_train_si55_cluster.tsv)",
    )
    ap.add_argument("--max-examples", type=int, default=5, help="Max overlap examples to print (default: 5)")
    args = ap.parse_args()

    # Load folds
    folds = _discover_folds(args.splits_dir, args.n_folds)

    # Print fold summary
    for line in fold_size_summary(folds):
        print(line)

    # Check fold disjointness
    ok1, msgs1 = check_fold_disjoint(folds, max_examples=args.max_examples)
    for m in msgs1:
        print(m)

    # Load clusters and check cluster-level leakage
    if not os.path.exists(args.cluster_tsv):
        print(f"[FAIL] cluster TSV not found: {args.cluster_tsv}")
        return 1

    rep2members = _load_cluster_tsv(args.cluster_tsv)
    ok2, msgs2 = check_cluster_spans_folds(folds, rep2members)
    for m in msgs2:
        print(m)

    # Final result
    if ok1 and ok2:
        print("[PASS] All leakage checks passed.")
        return 0
    else:
        print("[FAIL] Leakage checks failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())