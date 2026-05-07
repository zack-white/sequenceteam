#!/bin/tcsh

conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

#Set all tmp directories to shared project folder
setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache


# Goes into parent folder (/*_Trained/, /*_validation/)
# For every folder (/f*_Train/ and /f*_val/), create trim output folder.
# Trims the 3 columns (proteinID, GO:label, confidence) for every BP, CC, MF.


#!/bin/tcsh

conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

# Set tmp directories
setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache


# Base folder
set BASE=/r/proj167/lclark10/deepgo2/data/5_fold_Trained_50


# Loop through folds
foreach i (0 1 2 3 4)

    set DIR=$BASE/f${i}_Train_50
    echo "Processing $DIR"

    foreach TYPE (bp cc mf)

        set OLD=$DIR/f${i}_train_sequences_preds_${TYPE}.tsv.gz
        set NEW=$DIR/f${i}_train_50_sequences_preds_${TYPE}.tsv.gz

        if (-e $OLD) then
            echo "Renaming $OLD -> $NEW"
            mv $OLD $NEW
        else
            echo "Missing file: $OLD"
        endif

    end

end

echo "All training files renamed."