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

# Filter our default 0.1 threshold results to 0.5


mkdir -p CAFA_thr-50


foreach ont (mf cc bp)


    set infile = /r/proj167/lclark10/deepgo2/data/CAFA_Test/CAFA_thr-10/CAFA_trim_thr-10/testsuperset_unique_preds_${ont}.tsv.gz
    
    set outfile = /r/proj167/lclark10/deepgo2/data/CAFA_Test/CAFA_thr-50/CAFA_trim_thr-50/testsuperset_unique_preds_${ont}.tsv.gz

    echo "Filtering $infile"

    zcat $infile \
    | awk -F '\t' '$3 >= 0.5' \
    | gzip > $outfile


end

echo "Finished filtering threshold 0.5 predictions."