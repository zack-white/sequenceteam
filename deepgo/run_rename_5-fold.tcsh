#!/bin/tcsh

# Activate conda (REQUIRED for nohup scripts)
# source /path/to/conda.csh

# conda activate /r/proj167/lclark10/deepgo_env
# cd /r/proj167/lclark10/deepgo2

# #Set all tmp directories to shared project folder
# setenv TMPDIR /r/proj167/lclark10/tmp
# setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
# setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
# setenv TORCH_HOME /r/proj167/lclark10/torch_cache

# python -c "import torch; torch.cuda.empty_cache()"



set BASE=/r/proj167/lclark10/deepgo2/data/5_fold_val_50

foreach i (0 1 2 3 4)

    set DIR=$BASE/f${i}_val_50
    echo "Processing $DIR"

    foreach TYPE (bp cc mf)

        set OLD=$DIR/f${i}_val_sequences_preds_${TYPE}.tsv.gz
        set NEW=$DIR/f${i}_val_50_sequences_preds_${TYPE}.tsv.gz

        if (-e $OLD) then
            echo "Renaming $OLD -> $NEW"
            mv $OLD $NEW
        else
            echo "Missing $OLD"
        endif

    end

end

echo "Done renaming."