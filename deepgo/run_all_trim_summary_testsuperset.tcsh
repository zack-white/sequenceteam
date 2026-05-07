#!/bin/tcsh

conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

#Set all tmp directories to shared project folder
setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache


# Generates summary stats for every BP, CC, MF.

# Folder containing trimmed CAFA predictions
# Folder containing trimmed CAFA predictions
set BASE=/r/proj167/lclark10/deepgo2/data/CAFA_Test/CAFA_thr-50/CAFA_trim

echo "Processing CAFA trimmed predictions in $BASE"

foreach TYPE (bp cc mf)

    set FILE="$BASE/testsuperset_unique_preds_${TYPE}_trim.tsv.gz"
    set SUMMARY="$BASE/testsuperset_unique_preds_${TYPE}_trim_summary.txt"

    if (-e $FILE) then
        echo "Generating summary stats for $FILE"

        zcat $FILE | awk -v fname="$FILE" -f summary_stats.awk > $SUMMARY

    else
        echo "File $FILE not found, skipping $TYPE"
    endif

end

echo "CAFA TestSuperSet summary stats complete!"