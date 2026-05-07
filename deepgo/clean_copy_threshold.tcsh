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
# Source: make sure your environment is set up if needed
# cd to the parent folder containing 5_fold_Trained
cd /r/proj167/lclark10/deepgo2/data

# Create the new folder
set new_folder = "5_fold_val_50"
mkdir -p $new_folder

# Loop over each f[0-4]_Train folder
foreach d (`ls -d 5_fold_validation/f*_val`)
    
    # Extract folder name only (f0_Train, f1_Train, etc.)
    set base = `basename $d`
    
    # Skip any folder ending with _Train_trim (just in case)
    if ("$base" =~ "*_val_trim") then
        continue
    endif
    
    # Make corresponding folder in new_folder
    mkdir -p "$new_folder/${base}_50"
    
    # Copy all .fasta files
    foreach f (`ls $d/*.fasta`)
        cp $f "$new_folder/${base}_50/"
    end
end

echo "50 threshold copy complete. Only .fasta files copied into $new_folder."