#!/bin/bash

initdir=$1
input=$2
output=$3

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/etc/profile.d/conda.sh" ]; then
        . "/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/etc/profile.d/conda.sh"
    else
        export PATH="/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd $initdir
source scripts/setup.sh
echo python scripts/skim_ntuples_Hmm.py -i $input -o $output
python scripts/skim_ntuples_Hmm.py -i $input -o $output
        