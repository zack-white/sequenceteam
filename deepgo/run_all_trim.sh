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
set BASE=/r/proj167/lclark10/deepgo2/data/5_fold_Trained

# Loop over all f* folders
foreach FOLD (`ls -d $BASE/f[1]_Train`)
    echo "Processing folder: $FOLD"

    # Create trim folder
    set TRIM_FOLDER="$FOLD/`basename $FOLD`_trim"
    mkdir -p $TRIM_FOLDER

    # Loop over the 3 file types
    foreach TYPE (bp cc mf)
        # Construct input filename (adjust exact name if needed)
        set FILE="$FOLD/`basename $FOLD | tr '[:upper:]' '[:lower:]'`_sequences_preds_${TYPE}.tsv.gz"

        # Construct output filename
        set OUT="$TRIM_FOLDER/`basename $FOLD | tr '[:upper:]' '[:lower:]'`_sequences_preds_${TYPE}_trim.tsv.gz"

        # Check if input file exists
        if (-e $FILE) then
            echo "Trimming $FILE -> $OUT"
            zcat $FILE | awk '{print $1, $(NF-1), $NF}' | gzip > $OUT
        else
            echo "File $FILE not found, skipping $TYPE"
        endif
    end

end

echo "All folds trimmed successfully!"