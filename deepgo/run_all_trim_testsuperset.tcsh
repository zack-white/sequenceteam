#!/bin/tcsh

conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

#Set all tmp directories to shared project folder
setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache


# Goes into parent folder (/data/CAFA_Test/raw)

# Trims the 3 columns (proteinID, GO:label, confidence) for every BP, CC, MF. 
# Puts trimmed files in /CAFA_trim


# Base folder
set BASE=/r/proj167/lclark10/deepgo2/data/CAFA_Test/CAFA_thr-50

# Input / output folders
# set raw=$BASE
set CAFA_trim=$BASE/CAFA_trim

# File prefix
set FNAME=testsuperset_unique

# Create output directory if needed
mkdir -p $CAFA_trim

# Loop over BP / CC / MF
foreach TYPE (bp cc mf)

    set INFILE=$BASE/${FNAME}_preds_${TYPE}.tsv.gz
    set OUT=$CAFA_trim/${FNAME}_preds_${TYPE}_trim.tsv.gz

    if (-e $INFILE) then
        echo "Trimming $INFILE"
        zcat $INFILE | awk -F'\t' '{print $1 "\t" $(NF-1) "\t" $NF}' | gzip > $OUT
    else
        echo "Missing file: $INFILE"
    endif

end


echo "TestSuperSet trimmed successfully!"