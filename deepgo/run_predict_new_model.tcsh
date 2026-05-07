#!/bin/tcsh
# File: run_predict_new_model.tcsh
# Purpose: Loop over all Training folds and ontologies, generating DeepGO predictions

# ---- Activate environment and set caches ----
conda activate /r/proj167/lclark10/deepgo_env
cd /r/proj167/lclark10/deepgo2

setenv TMPDIR /r/proj167/lclark10/tmp
setenv PIP_CACHE_DIR /r/proj167/lclark10/.pip_cache
setenv PYTHONUSERBASE /r/proj167/lclark10/.local_python
setenv TORCH_HOME /r/proj167/lclark10/torch_cache

# ---- Accept arguments ----
if ($#argv < 2) then
    echo "Usage: tcsh run_predict_new_model.tcsh <base_dir> <device>"
    exit 1
endif

set base_dir = $1
set device   = $2

set training_dir = $base_dir/Training
set predict_script = /r/proj167/lclark10/deepgo2/predict_new_model.py
set batch_size = 32
set threshold = 0.1
set data_root = /r/proj167/lclark10/deepgo2/data

# ---- Loop over folds and ontologies ----
foreach fold (f0 f1 f2 f3 f4)
    echo "Processing fold $fold ..."
    foreach ont (bp cc mf)
        set fasta_file = $training_dir/$fold/${fold}_train_sequences.fasta
        if (-e $fasta_file) then
            set log_file = output/${fold}_train_${ont}_predictions_`date +%Y%m%d_%H%M%S`.log
            echo "Submitting $ont predictions for $fold -> logging to $log_file"

            # Run predict.py in background via nohup
            nohup python $predict_script \
                -if $fasta_file \
                -d $device \
                -bs $batch_size \
                -t $threshold \
                -dr $data_root \
                >& $log_file &
        else
            echo "FASTA file not found: $fasta_file"
        endif
    end
end

echo "All Training predictions submitted."