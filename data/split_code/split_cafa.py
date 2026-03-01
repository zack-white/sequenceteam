from collections import defaultdict
import random

K = 5
SEED = 42
cluster_tsv = "mmseqs_out/cafa_train_si55_cluster.tsv"

rep2members = defaultdict(list)

with open(cluster_tsv) as f:
    for line in f:
        rep, mem = line.rstrip("\n").split("\t")
        rep2members[rep].append(mem)

clusters = sorted(rep2members.items(), key=lambda x: len(x[1]), reverse=True)

random.seed(SEED)

fold_ids = [set() for _ in range(K)]
fold_sizes = [0]*K

for rep, members in clusters:
    # assign to smallest fold
    j = min(range(K), key=lambda i: fold_sizes[i])
    for m in members:
        fold_ids[j].add(m)
    fold_sizes[j] += len(members)

for i in range(K):
    with open(f"fold{i}_ids.txt", "w") as f:
        for sid in sorted(fold_ids[i]):
            f.write(sid + "\n")

print("Fold sizes:", fold_sizes)