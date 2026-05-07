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


# Source directory (current directory)
set SRC="."

# Destination directories
set TRAIN_DEST="./data/5_fold_Trained_50"
set VAL_DEST="./data/5_fold_val_50"

echo "Moving train prediction files..."
foreach f (${SRC}/*train_*.tsv.gz)
    if (-e "$f") then
        echo "Moving $f -> $TRAIN_DEST/"
        mv "$f" "$TRAIN_DEST/"
    endif
end

echo "Moving validation prediction files..."
foreach f (${SRC}/*val_*.tsv.gz)
    if (-e "$f") then
        echo "Moving $f -> $VAL_DEST/"
        mv "$f" "$VAL_DEST/"
    endif
end

echo "Done."