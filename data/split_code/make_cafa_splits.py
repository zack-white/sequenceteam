#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple


def read_id_list(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 只取第一列，防止多余空格
            ids.append(s.split()[0])
    return ids


def write_id_list(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in ids:
            f.write(x + "\n")


def load_keep_set(path: Path) -> Set[str]:
    return set(read_id_list(path))


def filter_tsv_by_first_col(
    in_path: Path,
    out_path: Path,
    keep: Set[str],
    keep_header: bool = True,
) -> Tuple[int, int]:
    """
    假设 TSV 第一列是 protein id
    返回：(输出行数(不含header), 总扫描行数(不含header))
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    n_total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        first_line = fin.readline()
        if not first_line:
            return 0, 0

        # 粗暴判断 header：如果第一列不是典型蛋白ID且包含字母数字下划线等，可能是header
        # 这里按用户需求：默认保留第一行当 header
        if keep_header:
            fout.write(first_line)
        else:
            # 如果不保留 header，把它当普通行处理
            parts = first_line.rstrip("\n").split("\t")
            if parts and parts[0] in keep:
                fout.write(first_line)
                n_out += 1
            n_total += 1

        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            pid = parts[0]
            if pid in keep:
                fout.write(line)
                n_out += 1

    return n_out, n_total


def filter_fasta_by_ids(
    fasta_in: Path,
    fasta_out: Path,
    keep: Set[str],
) -> Tuple[int, int]:
    """
    假设 FASTA header 行：>P12345 blahblah
    取第一个 token 去掉 '>' 当 protein id
    返回：(输出序列条数, 扫描到的总序列条数)
    """
    fasta_out.parent.mkdir(parents=True, exist_ok=True)

    n_out = 0
    n_total = 0
    writing = False

    with fasta_in.open("r", encoding="utf-8", errors="replace") as fin, fasta_out.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith(">"):
                n_total += 1
                # 取第一段 token
                head = line[1:].strip()
                pid = head.split()[0]
                writing = pid in keep
                if writing:
                    n_out += 1
                    fout.write(line)
            else:
                if writing:
                    fout.write(line)

    return n_out, n_total


def check_overlap(a: Set[str], b: Set[str]) -> List[str]:
    inter = sorted(a.intersection(b))
    return inter[:20]  # 只返回前 20 个，避免炸屏


def main():
    p = argparse.ArgumentParser(
        description="Make CAFA5 cluster-level K-fold splits and export FASTA/TSV subsets."
    )
    p.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    p.add_argument(
        "--fold-prefix",
        type=str,
        default="fold",
        help="Fold file prefix: fold0_ids.txt ... (default: fold)",
    )
    p.add_argument(
        "--fold-suffix",
        type=str,
        default="_ids.txt",
        help="Fold file suffix (default: _ids.txt)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="splits",
        help="Output directory (default: splits/)",
    )
    p.add_argument(
        "--train-fasta",
        type=str,
        default="cafa-5-protein-function-prediction/Train/train_sequences.fasta",
        help="Path to train_sequences.fasta",
    )
    p.add_argument(
        "--train-terms",
        type=str,
        default="cafa-5-protein-function-prediction/Train/train_terms.tsv",
        help="Path to train_terms.tsv",
    )
    p.add_argument(
        "--train-taxonomy",
        type=str,
        default="cafa-5-protein-function-prediction/Train/train_taxonomy.tsv",
        help="Path to train_taxonomy.tsv",
    )
    p.add_argument(
        "--keep-header",
        action="store_true",
        help="Keep first line as header for TSV outputs (recommended).",
    )
    args = p.parse_args()

    k = args.k
    outdir = Path(args.outdir)
    fasta_in = Path(args.train_fasta)
    terms_in = Path(args.train_terms)
    tax_in = Path(args.train_taxonomy)

    # 读所有 fold ids
    fold_lists: List[List[str]] = []
    for i in range(k):
        fp = Path(f"{args.fold_prefix}{i}{args.fold_suffix}")
        if not fp.exists():
            raise FileNotFoundError(f"Missing fold file: {fp}")
        fold_lists.append(read_id_list(fp))

    fold_sets = [set(x) for x in fold_lists]

    # 先检查：fold 之间有无重叠（理论上不该有）
    for i in range(k):
        for j in range(i + 1, k):
            inter = fold_sets[i].intersection(fold_sets[j])
            if inter:
                sample = sorted(list(inter))[:10]
                raise RuntimeError(
                    f"Leakage detected: fold{i} and fold{j} overlap (e.g., {sample})"
                )

    print(f"[OK] Loaded {k} folds. No overlap across folds.")

    # 逐 fold 生成 train/val & 切文件
    for i in range(k):
        fold_dir = outdir / f"fold{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_ids = sorted(fold_sets[i])
        train_ids = sorted(set().union(*[fold_sets[j] for j in range(k) if j != i]))

        # 再次检查 train/val overlap
        overlap = check_overlap(set(train_ids), set(val_ids))
        if overlap:
            raise RuntimeError(f"Train/Val overlap in fold{i}: {overlap}")

        write_id_list(fold_dir / "train_ids.txt", train_ids)
        write_id_list(fold_dir / "val_ids.txt", val_ids)

        train_set = set(train_ids)
        val_set = set(val_ids)

        # TSV: terms
        train_terms_out = fold_dir / "train_terms.tsv"
        val_terms_out = fold_dir / "val_terms.tsv"
        n_train_terms, n_total_terms = filter_tsv_by_first_col(
            terms_in, train_terms_out, train_set, keep_header=args.keep_header
        )
        n_val_terms, _ = filter_tsv_by_first_col(
            terms_in, val_terms_out, val_set, keep_header=args.keep_header
        )

        # TSV: taxonomy
        train_tax_out = fold_dir / "train_taxonomy.tsv"
        val_tax_out = fold_dir / "val_taxonomy.tsv"
        n_train_tax, n_total_tax = filter_tsv_by_first_col(
            tax_in, train_tax_out, train_set, keep_header=args.keep_header
        )
        n_val_tax, _ = filter_tsv_by_first_col(
            tax_in, val_tax_out, val_set, keep_header=args.keep_header
        )

        # FASTA
        train_fasta_out = fold_dir / "train_sequences.fasta"
        val_fasta_out = fold_dir / "val_sequences.fasta"
        n_train_fa, n_total_fa = filter_fasta_by_ids(fasta_in, train_fasta_out, train_set)
        n_val_fa, _ = filter_fasta_by_ids(fasta_in, val_fasta_out, val_set)

        # 打印汇总
        print(
            f"[fold{i}] ids: train={len(train_ids)} val={len(val_ids)} | "
            f"FASTA: train={n_train_fa}/{n_total_fa} val={n_val_fa}/{n_total_fa} | "
            f"terms(lines excl header): train={n_train_terms}/{n_total_terms} val={n_val_terms}/{n_total_terms} | "
            f"taxonomy(lines excl header): train={n_train_tax}/{n_total_tax} val={n_val_tax}/{n_total_tax}"
        )

    print(f"\nDone. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()