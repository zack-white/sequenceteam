Sequence Train-Validation 5-Fold Split
Author: Yihan Fang
Last Updated: Feb 26th
Sequence Team: Jess Charalel, Lisa Clark, Yihan Fang, Angela Huynh, Rowan Macy, Amanda Pang, Zachary White 


Homology Clustering - Completed the week of 9th
We first cluster similar homologous proteins using MMseqs2.

Dataset
CAFA-5 Kaggle dataset
https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data

Logs

Code 
mmseqs easy-cluster
--min-seq-id 0.55
-c 0.8
--cov-mode 1

Tools Used/ Environment
Standard Mac terminal
doi: 10.1038/nbt.3988

Results 
142, 246 sequences
~ 70,452 clusters
~ 70k representatives

WHY NOT RANDOM SPLIT
Random splitting of protein sequences leads to homology leakage, where highly similar sequences appear in both train and test sets, artificially inflating model performance.
Justification & Explanation
Breakdown of Parameters
Sequence Identity >= 55%
A 55% sequence identity threshold removes close homologs while retaining moderate evolutionary diversity. It prevents trivial near-duplicate leakage while preserving sufficient data for learning. 
Alignment covering >=80% sequences
Coverage based on target (cover mode 1)
One-Sentence Summary
Highly similar homologous proteins are grouped together.

Cluster-level 5-fold Split - Done the week of 16th, Completed the week of 23rd
After clustering, splitting at the cluster level is performed to prevent homology leakage across folds.

Procedure
Each MMseqs cluster is treated as an individual unit.
Clusters are sorted by size.
Clusters are greedily assigned to the fold with the smallest current size.
Random seed fixed at 42 for reproducibility.
Code
See the provided files.

Verification 
Verification Includes:
	Mutual exclusivity of fold ID sets
	Cluster-level non-overlap validation (70,452 clusters checked)
	All checks pass

Run Output:
	[INFO] Fold sizes (val IDs):
  - fold0: 28,450  (splits/fold0/val_ids.txt)
  - fold1: 28,449  (splits/fold1/val_ids.txt)
  - fold2: 28,449  (splits/fold2/val_ids.txt)
  - fold3: 28,449  (splits/fold3/val_ids.txt)
  - fold4: 28,449  (splits/fold4/val_ids.txt)
[INFO] Size stats: mean=28449.20, std=0.40, min=28449, max=28450
[OK] Fold ID sets are mutually exclusive (no ID appears in multiple folds).
[OK] No MMseqs cluster spans multiple folds (cluster-level split prevents leakage).
[INFO] Clusters checked: 70452
[PASS] All leakage checks passed.
Fold Statistics
Mean = 28449.20
Std = 0.40

Why Cluster-Level but not Sequence-Level? (Again!)
If sequences were split individually, members of the same cluster could appear in different folds, resulting in >= 55% identity leakage.

One-Sentence Summary
No MMseqs cluster spans multiple folds; therefore, no ≥55% identity homologous leakage exists across train and validation splits.

