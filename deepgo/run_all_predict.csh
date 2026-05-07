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

foreach f (data/5_fold_validation/f[2-4]_val/*fasta)

    # Extract filename (e.g., f2_val_sequences.fasta)
    set fname = `basename $f`

    # Extract prefix (e.g., f2)
    set prefix = `echo $fname | cut -d'_' -f1`

    # echo "Running $prefix on $f"

    nohup python predict.py \
        -if $f \
        -d cuda:2 \
        >& output/${prefix}_validation_predictions.log &
    
    # Finish this job before starting next fold
    wait

end