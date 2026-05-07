#!/bin/tcsh

# Activate conda (REQUIRED for nohup scripts)
# source /path/to/conda.csh

conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

#Set all tmp directories to shared project folder
setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache

python -c "import torch; torch.cuda.empty_cache()"
#Run predict.py on every *.fasta file in the 5-fold folders f1-f4
# excluded f0 bc ran that manually

# Fold 2
nohup python predict.py \
    -if data/5_fold_validation/f2_val/f2_val_sequences.fasta \
    -d cuda:2 \
    >& output/f2_validation_predictions.log
set pid=$!
echo "Started f2 (PID $pid)"
wait
python -c "import torch; import gc; gc.collect(); import torch; torch.cuda.empty_cache()"
sleep 5
echo "Finished f2 (PID $pid)"

# # Fold 3
# nohup python predict.py \
#     -if data/5_fold_validation/f3_val/f3_val_sequences.fasta \
#     -d cuda:2 \
#     >& output/f3_validation_predictions.log &
# set pid=$!
# echo "Started f3 (PID $pid)"
# wait $pid
# python -c "import torch; import gc; gc.collect(); import torch; torch.cuda.empty_cache()"
# sleep 5


# # Fold 4
# nohup python predict.py \
#     -if data/5_fold_validation/f4_val/f4_val_sequences.fasta \
#     -d cuda:2 \
#     >& output/f4_validation_predictions.log &
# set pid=$!
# echo "Started f4 (PID $pid)"
# wait $pid
# python -c "import torch; import gc; gc.collect(); import torch; torch.cuda.empty_cache()"
# sleep 5

# echo "All folds 2-4 completed!"