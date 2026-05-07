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


# Base folder
set BASE=/r/proj167/lclark10/deepgo2/data/5_fold_train_50


# Loop over all fold[0-4] directories
foreach FOLD ($BASE/f[0-4]_train_50)

    set FNAME=`basename $FOLD`
    echo "Processing folder: $FNAME"

    # Create trim directory
    set TRIM_FOLDER=$FOLD/${FNAME}_trim
    mkdir -p $TRIM_FOLDER


    # Loop over BP / CC / MF
    foreach TYPE (bp cc mf)

        set FILE=$FOLD/${FNAME}_sequences_preds_${TYPE}.tsv.gz
        set OUT=$TRIM_FOLDER/${FNAME}_sequences_preds_${TYPE}_trim.tsv.gz

        if (-e $FILE) then
            echo "Trimming $FILE"
            zcat $FILE | awk '{print $1, $(NF-1), $NF}' | gzip > $OUT
        else
            echo "Missing file: $FILE"
        endif

    end

end

echo "All folds trimmed successfully!"