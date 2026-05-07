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
#Run predict.py on every *.fasta file in the 5-fold folders f0-f4
# Tested on f0


# foreach f (data/5_fold_validation/f[0-4]_val/*fasta)
foreach f (data/5_fold_val_50/f[0-5]_val_50/*.fasta)

    # Extract filename (e.g., f2_val_sequences.fasta)
    set fname = `basename $f`

    # Extract prefix (e.g., f2)
    set prefix = `echo $fname | cut -d'_' -f1`

    echo "Running $prefix on $f"

    python predict_50.py \
        -if $f \
        -d cuda:1 \
        >& output/${prefix}_val_predictions_thr-50.log
    
    # # Finish this job before starting next fold
    # wait

end