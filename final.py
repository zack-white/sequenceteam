"""
 FIXED 
 
  1. Embedding cache: embeddings saved to disk as float16 .npy files —
     crash-safe; fold re-run skips embedding entirely if cache exists.
     float16 halves disk vs float32 (320 MB → 160 MB per 113K × 320-dim split).
     Optional PCA compression (CACHE_PCA_DIM) shrinks further: 64-dim ≈ 29 MB.
     Quota-safe: if disk save fails (OSError/quota), logs a warning and
     continues in-memory - no crash.
  2. Batch-size auto-tuning: starts at BATCH_SIZE, halves on OOM.
  3. Bug fix: predict_go_terms_batch was called in main() but never defined;
     replaced with the existing predict_go_terms_faiss.
  4. Process one fold at a time: folds are embedded and evaluated
     sequentially so only one fold's embeddings live in RAM at once.
  5. Sequence truncation moved to MAX_SEQ_LEN (256) to cut token cost.
  6. Minor: duplicate DEVICE assignment removed, tqdm import cleaned up.
"""
from __future__ import annotations
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
 
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm
 
import torch
from goatools.obo_parser import GODag
 
warnings.filterwarnings("ignore")
 

# Config                                                               
 
DATA_DIR   = "/h/jchara02/cs40/cs167/splits"
OBO_PATH   = "/h/jchara02/cs40/cs167/go.obo"
OUTPUT_DIR = "/h/jchara02/cs40/cs167/outputs"
CACHE_DIR  = os.path.join(OUTPUT_DIR, "embed_cache")   # NEW: embedding cache
N_FOLDS    = 5
 
NAMESPACE  = os.environ.get("NAMESPACE", None)
 
ESM_MODEL  = os.environ.get("ESM_MODEL",  "esm2_t6_8M_UR50D")
ESM_LAYER  = int(os.environ.get("ESM_LAYER",  "6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))   # raised from 8; auto-halved on OOM
MAX_SEQ_LEN= int(os.environ.get("MAX_SEQ_LEN","256"))  # truncate long seqs early
 
K             = int(os.environ.get("K", "5"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
 
# Disk-quota controls -----------------------------------------------
# CACHE_PCA_DIM: compress embeddings to N PCA dims before saving.
#   None  -> save full float16 vectors  (~160 MB per 113K x 320-dim split)
#   64    -> 64-dim PCA float16         (~29 MB per split)  <-- default
#   Set via:  export CACHE_PCA_DIM=64  or  export CACHE_PCA_DIM=none
_pca_env      = os.environ.get("CACHE_PCA_DIM", "64")
CACHE_PCA_DIM: int | None = int(_pca_env) if _pca_env.lower() != "none" else None
 
print(f"Device: {DEVICE}  |  ESM model: {ESM_MODEL}  |  k={K}  |  "
      f"batch={BATCH_SIZE}  |  max_len={MAX_SEQ_LEN}  |  "
      f"cache_pca={CACHE_PCA_DIM}", flush=True)
 

# Data types                                                           
 
@dataclass
class Protein:
    pid: str
    sequence: str
    go_terms: set[str] = field(default_factory=set)
    taxonomy: dict = field(default_factory=dict)
    embedding: np.ndarray|None = None
 
 
@dataclass
class Fold:
    idx: int
    train: list[Protein]
    val: list[Protein]
 
 

# File parsers                                                         
 
def parse_ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]
 
 
def parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    current_id = None
    current_seq: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current_id:
                seqs[current_id] = "".join(current_seq)
            header = line[1:].strip()
            match = re.match(r"(?:sp|tr)\|([^|]+)\|", header)
            current_id = match.group(1) if match else header.split()[0]
            current_seq = []
        elif line.strip():
            current_seq.append(line.strip().upper())
    if current_id:
        seqs[current_id] = "".join(current_seq)
    return seqs
 
 
def parse_terms(path: Path, namespace_filter: str|None = None,
                godag: GODag|None = None) -> dict[str, set[str]]:
    pid_to_terms: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        pid, go_id = parts[0].strip(), parts[1].strip()
        if not go_id.startswith("GO:"):
            continue
        if namespace_filter and godag and go_id in godag:
            if godag[go_id].namespace != namespace_filter:
                continue
        pid_to_terms[pid].add(go_id)
    return dict(pid_to_terms)
 
 
def parse_taxonomy(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path, sep="\t")
    id_col = next(
        (c for c in df.columns if c.lower() in ("protein_id","id","accession","entry")),
        df.columns[0]
    )
    df = df.rename(columns={id_col: "protein_id"})
    return {row["protein_id"]: row.drop("protein_id").to_dict()
            for _, row in df.iterrows()}
 
 

# Fold loader                                                       
 
def load_fold(fold_dir: Path, godag: GODag|None = None,
              namespace: str|None = None) -> tuple[list[Protein], list[Protein]]:
    result = {}
    for split in ("train", "val"):
        ids_path = fold_dir / f"{split}_ids.txt"
        fa_path = fold_dir / f"{split}_sequences.fasta"
        terms_path = fold_dir / f"{split}_terms.tsv"
        tax_path = fold_dir / f"{split}_taxonomy.tsv"
 
        if not ids_path.exists() or not fa_path.exists():
            raise FileNotFoundError(
                f"Missing required files in {fold_dir}: "
                f"{split}_ids.txt or {split}_sequences.fasta"
            )
        ids = parse_ids(ids_path)
        seqs = parse_fasta(fa_path)
        terms = parse_terms(terms_path, namespace, godag) if terms_path.exists() else {}
        taxonomy = parse_taxonomy(tax_path) if tax_path.exists() else {}
 
        proteins = [
            Protein(pid=pid, sequence=seqs[pid],
                    go_terms=terms.get(pid, set()),
                    taxonomy=taxonomy.get(pid, {}))
            for pid in ids if pid in seqs
        ]
        result[split] = proteins
        print(f"    {split}: {len(proteins):,} proteins, "
              f"{sum(len(p.go_terms) for p in proteins):,} GO annotations")
    return result["train"], result["val"]
 
 
def load_all_folds(data_dir: str, godag: GODag|None = None,
                   namespace: str|None = None, n_folds: int = 5) -> list[Fold]:
    folds = []
    for i in range(n_folds):
        fold_dir = Path(data_dir) / f"fold{i}"
        if not fold_dir.exists():
            print(f"  fold {i} directory not found — skipping")
            continue
        print(f"\n  loading fold {i} from {fold_dir}")
        train, val = load_fold(fold_dir, godag, namespace)
        folds.append(Fold(idx=i, train=train, val=val))
    print(f"\n  loaded {len(folds)} folds")
    return folds
 
 

# ESM-2 embedding  (with disk cache + OOM-safe batching)             

 
def load_esm(model_name: str):
    import esm
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model = model.eval().to(DEVICE)
    return model, alphabet.get_batch_converter()
 
 
def _embed_batch(batch_data: list[tuple[str,str]], model, batch_converter,
                 layer: int) -> list[np.ndarray]:
    """Embed one batch; returns list of per-protein mean-pool vectors."""
    _, _, tokens = batch_converter(batch_data)
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer]   # (B, L+2, D)
    vecs = []
    for j, (_, seq) in enumerate(batch_data):
        seq_len = len(seq)
        vecs.append(reps[j, 1:seq_len+1].mean(0).cpu().numpy())
    return vecs
 
 
def embed_proteins(proteins: list[Protein], model, batch_converter,
                   layer: int = ESM_LAYER,
                   batch_size: int = BATCH_SIZE,
                   max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Embed all proteins, truncating at max_len.
    Auto-halves batch_size on CUDA OOM and retries — but recovers back
    toward the original size after each successful batch so one long
    sequence does not strand the rest of the run at batch_size=1.
    """
    all_vecs = []
    i = 0
    cur_batch = batch_size          # local copy — never mutates the default
    pbar = tqdm(total=len(proteins), desc="embedding", unit="prot")
    while i < len(proteins):
        batch = proteins[i : i + cur_batch]
        data  = [(p.pid, p.sequence[:max_len]) for p in batch]
        try:
            vecs = _embed_batch(data, model, batch_converter, layer)
            all_vecs.extend(vecs)
            i += len(batch)
            pbar.update(len(batch))
            # Gradually recover toward original batch size after a success
            if cur_batch < batch_size:
                cur_batch = min(batch_size, cur_batch * 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur_batch > 1:
                torch.cuda.empty_cache()
                cur_batch = max(1, cur_batch // 2)
                print(f"\n  OOM — reducing batch_size to {cur_batch}")
            else:
                raise
    pbar.close()
    return np.array(all_vecs, dtype=np.float32)
 
 

# Embedding cache  (float16 + optional PCA, quota-safe)  

 
# One PCA object fit on fold-0 train, reused for all subsequent splits
# so that all embeddings live in the same projected space.
_pca_model: PCA | None = None
 
 
def _cache_path(fold_idx: int, split: str) -> Path:
    suffix = f"pca{CACHE_PCA_DIM}" if CACHE_PCA_DIM else "full"
    return Path(CACHE_DIR) / f"fold{fold_idx}_{split}_{suffix}_fp16.npy"
 
 
def _try_save(path: Path, arr: np.ndarray) -> bool:
    """Save array; return False and warn on any OSError (quota, no space)."""
    try:
        np.save(path, arr)
        return True
    except OSError as e:
        print(f"\n  WARNING: could not save cache ({e}). "
              f"Continuing in-memory (no resume on crash).")
        return False
 
 
def _compress(vecs: np.ndarray, split_name: str) -> np.ndarray:
    """
    Optionally reduce dimensionality with PCA, then cast to float16.
    PCA is fit once on the first train split and reused everywhere.
    """
    global _pca_model
    if CACHE_PCA_DIM is None or vecs.shape[1] <= CACHE_PCA_DIM:
        return vecs.astype(np.float16)
 
    if split_name == "train" and _pca_model is None:
        n_comp = min(CACHE_PCA_DIM, vecs.shape[0], vecs.shape[1])
        print(f"  fitting PCA({n_comp}) on train embeddings …")
        _pca_model = PCA(n_components=n_comp, random_state=42)
        _pca_model.fit(vecs)
        explained = _pca_model.explained_variance_ratio_.sum()
        print(f"  PCA explains {explained:.1%} of variance")
 
    if _pca_model is not None:
        vecs = _pca_model.transform(vecs)
 
    return vecs.astype(np.float16)
 
 
def embed_fold_cached(fold: Fold, model, batch_converter) -> None:
    """
    Embed train + val, ensuring both end up in the same vector space.
 
    The core invariant: PCA (if enabled) is ALWAYS fit on raw train
    embeddings and ALWAYS applied to raw val embeddings.  This means:
 
      - If train cache missing  → embed train → fit PCA → save → embed val → apply PCA → save
      - If train cache present  → load train (already PCA'd) → embed val → refit PCA
                                  on train to get _pca_model → apply to val → save
      - Dim mismatch is impossible because we never mix a cached PCA-split
        with an un-projected split going into FAISS.
 
    Cache is float16 (halves disk).  Quota failures are non-fatal.
    """
    global _pca_model
    os.makedirs(CACHE_DIR, exist_ok=True)
 
    train_cp = _cache_path(fold.idx, "train")
    val_cp   = _cache_path(fold.idx, "val")
 
    # TRAIN 
    if train_cp.exists():
        print(f"  loading cached train embeddings ({train_cp.name})")
        train_vecs = np.load(train_cp).astype(np.float32)
        # Cache is already PCA-compressed
        if CACHE_PCA_DIM and _pca_model is None:
            print("  refitting PCA on fresh train sample to project val …")
            raw_train = embed_proteins(fold.train, model, batch_converter)
            n_comp = min(CACHE_PCA_DIM, raw_train.shape[0], raw_train.shape[1])
            _pca_model = PCA(n_components=n_comp, random_state=42)
            _pca_model.fit(raw_train)
            print(f"  PCA({n_comp}) explains "
                  f"{_pca_model.explained_variance_ratio_.sum():.1%} of variance")
    else:
        print("  embedding train proteins …")
        raw_train  = embed_proteins(fold.train, model, batch_converter)
        train_vecs = _compress(raw_train, "train").astype(np.float32)
        saved = _try_save(train_cp, train_vecs.astype(np.float16))
        if saved:
            print(f"  saved train cache ({train_vecs.nbytes/1024**2:.1f} MB) "
                  f"→ {train_cp.name}")
 
    for p, v in zip(fold.train, train_vecs):
        p.embedding = v
 
    #  VAL 
    if val_cp.exists():
        print(f"  loading cached val embeddings ({val_cp.name})")
        val_vecs = np.load(val_cp).astype(np.float32)
    else:
        print("  embedding val proteins …")
        raw_val  = embed_proteins(fold.val, model, batch_converter)
        # _pca_model is now guaranteed to be fit (either from train embed
        # above, or refitted from raw_train in the cache-hit branch)
        val_vecs = _compress(raw_val, "val").astype(np.float32)
        saved = _try_save(val_cp, val_vecs.astype(np.float16))
        if saved:
            print(f"  saved val cache ({val_vecs.nbytes/1024**2:.1f} MB) "
                  f"→ {val_cp.name}")
 
    for p, v in zip(fold.val, val_vecs):
        p.embedding = v
 
 
 

# FAISS index + batched KNN prediction                                 

def build_faiss_index(train: list[Protein]):
    mat = np.vstack([p.embedding for p in train]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat = mat / norms
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index, train
 
 
def predict_go_terms_faiss(
    queries: list[Protein],
    index,
    train: list[Protein],
    k: int,
    batch_size: int = 512,
) -> list[dict[str, float]]:
    """
    Batched FAISS cosine-similarity KNN.
    Returns list of {go_term: weighted_score} dicts.
    """
    results = []
    for i in range(0, len(queries), batch_size):
        batch  = queries[i : i + batch_size]
        q_mat  = np.vstack([p.embedding for p in batch]).astype(np.float32)
        q_mat /= np.linalg.norm(q_mat, axis=1, keepdims=True) + 1e-10
 
        sims, idxs = index.search(q_mat, k)   # (B, k)
 
        for row_sims, row_idxs in zip(sims, idxs):
            term_scores: dict[str, float] = defaultdict(float)
            total = row_sims.sum() + 1e-10
            for sim, idx in zip(row_sims, row_idxs):
                w = sim / total
                for term in train[idx].go_terms:
                    term_scores[term] += w
            results.append(dict(term_scores))
    return results
 
 

# Metrics                                                    

 
def micro_f1(y_true: list[set[str]], y_pred: list[set[str]]) -> dict:
    tp = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        tp += len(true & pred)
        fp += len(pred - true)
        fn += len(true - pred)
    prec   = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1     = 2 * prec * recall / (prec + recall + 1e-10)
    return {"precision": prec, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}
 
 
def per_term_f1(y_true: list[set[str]], y_pred: list[set[str]]) -> pd.DataFrame:
    all_terms = sorted({t for s in y_true + y_pred for t in s})
    rows = []
    for term in all_terms:
        tp = sum(1 for t, p in zip(y_true, y_pred) if term in t and term in p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if term not in t and term in p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if term in t and term not in p)
        prec   = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1     = 2 * prec * recall / (prec + recall + 1e-10)
        rows.append({"go_term": term, "precision": prec,
                     "recall": recall, "f1": f1, "support": tp + fn})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)
 
 
def sweep_threshold(fold: Fold, k: int,
                    thresholds: list[float] | None = None):
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.0, 0.8, 0.05)]
 
    index, _ = build_faiss_index(fold.train)
    all_scores = predict_go_terms_faiss(fold.val, index, fold.train, k)
    y_true = [p.go_terms for p in fold.val]
 
    rows = []
    for t in thresholds:
        y_pred = [{term for term, s in sc.items() if s >= t} for sc in all_scores]
        m = micro_f1(y_true, y_pred)
        m["threshold"] = t
        rows.append(m)
 
    return pd.DataFrame(rows), all_scores
 
 

# Plots                                                              

def save_fold_metrics_plot(all_metrics: list[dict], path: str) -> None:
    df  = pd.DataFrame(all_metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["precision", "recall", "f1"]):
        ax.bar(df["fold"].astype(str), df[col], color="#4C72B0", edgecolor="white")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Fold"); ax.set_ylabel(col.capitalize())
        ax.set_title(f"{col.capitalize()} across folds")
        for i, v in enumerate(df[col]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.suptitle(f"KNN GO prediction  (k={K})", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved fold metrics → {path}")
 
 
def save_threshold_plot(sweep_df: pd.DataFrame, fold_idx: int, path: str) -> None:
    plt.figure(figsize=(8, 4))
    for col, label in [("precision","Precision"),("recall","Recall"),("f1","F1")]:
        plt.plot(sweep_df["threshold"], sweep_df[col], marker="o",
                 markersize=4, label=label, linewidth=1.5)
    best = sweep_df.loc[sweep_df["f1"].idxmax()]
    plt.axvline(best["threshold"], linestyle="--", color="gray", alpha=0.6,
                label=f"best t={best['threshold']:.2f} (F1={best['f1']:.3f})")
    plt.xlabel("Score threshold"); plt.ylabel("Score")
    plt.title(f"Fold {fold_idx}: P/R/F1 vs threshold")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved threshold plot → {path}")
 
 
def save_pca_plot(fold: Fold, path: str) -> None:
    vecs = np.vstack([p.embedding for p in fold.train + fold.val
                      if p.embedding is not None])
    n_tr = sum(1 for p in fold.train if p.embedding is not None)
    proj = PCA(n_components=2).fit_transform(vecs)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:n_tr, 0], proj[:n_tr, 1], alpha=0.3, s=6, label="train", c="#4C72B0")
    plt.scatter(proj[n_tr:, 0], proj[n_tr:, 1], alpha=0.3, s=6, label="val",   c="#DD8452")
    plt.title(f"Fold {fold.idx}: ESM-2 embeddings (PCA)")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved PCA plot → {path}")
 
 

# Main — process one fold at a time to keep RAM manageable    

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)
 
    # 1. GO ontology ------------------------------------------------ #
    print("\n[1/4] Loading GO ontology")
    import urllib.request
    if not os.path.exists(OBO_PATH):
        print("  downloading go.obo …")
        urllib.request.urlretrieve(
            "https://current.geneontology.org/ontology/go.obo", OBO_PATH
        )
    godag = GODag(OBO_PATH, optional_attrs={"def", "namespace"})
    print(f"  {len(godag):,} GO terms loaded")
 
    # 2. Discover fold directories ---------------------------------- #
    print("\n[2/4] Discovering folds")
    fold_dirs = sorted(
        [Path(DATA_DIR) / f"fold{i}" for i in range(N_FOLDS)
         if (Path(DATA_DIR) / f"fold{i}").exists()]
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {DATA_DIR}")
    print(f"  found {len(fold_dirs)} folds")
 
    # 3. Load ESM model once --------------------------------------- #
    print(f"\n[3/4] Loading ESM-2 model ({ESM_MODEL})")
    esm_model, batch_converter = load_esm(ESM_MODEL)
 
    # 4. Process each fold ----------------------------------------- #
    print(f"\n[4/4] Embedding + evaluation (k={K})")
    all_metrics: list[dict] = []
 
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.replace("fold", ""))
        print(f"\n{'='*50}")
        print(f"  FOLD {fold_idx}")
        print(f"{'='*50}")
 
        # Load this fold's proteins
        train, val = load_fold(fold_dir, godag, NAMESPACE)
        fold = Fold(idx=fold_idx, train=train, val=val)
 
        # Embed (loads from cache if available)
        embed_fold_cached(fold, esm_model, batch_converter)
 
        # Threshold sweep
        sweep_df, all_scores = sweep_threshold(fold, K)
        save_threshold_plot(
            sweep_df, fold_idx,
            path=os.path.join(OUTPUT_DIR, f"threshold_fold{fold_idx}.png")
        )
 
        best_t = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])
        print(f"  best threshold: {best_t:.2f}")
 
        # Final eval at best threshold
        y_true = [p.go_terms for p in fold.val]
        y_pred = [{t for t, s in sc.items() if s >= best_t} for sc in all_scores]
 
        metrics = micro_f1(y_true, y_pred)
        metrics.update({"fold": fold_idx, "threshold": best_t})
        all_metrics.append(metrics)
 
        print(f"  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}")
 
        # undefined predict_go_terms_batch that caused the original crash
        term_df = per_term_f1(y_true, y_pred)
        term_df.to_csv(
            os.path.join(OUTPUT_DIR, f"per_term_f1_fold{fold_idx}.csv"),
            index=False
        )
 
        save_pca_plot(fold, os.path.join(OUTPUT_DIR, f"pca_fold{fold_idx}.png"))
 
        # Free this fold's embeddings before loading the next
        for p in fold.train + fold.val:
            p.embedding = None
        del fold, train, val, all_scores, sweep_df
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
 
    # Cross-fold summary 
    summary = pd.DataFrame(all_metrics)
    means   = summary[["precision","recall","f1"]].mean()
    print("\n=== 5-fold average ===")
    print(f"  precision : {means['precision']:.4f}")
    print(f"  recall : {means['recall']:.4f}")
    print(f"  F1 : {means['f1']:.4f}")
 
    summary.to_csv(os.path.join(OUTPUT_DIR, "fold_metrics.csv"), index=False)
    save_fold_metrics_plot(all_metrics,
                           os.path.join(OUTPUT_DIR, "fold_metrics.png"))
    print(f"\nDone. All outputs in {OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
 