#!/usr/bin/env python

import os
from datetime import datetime
from argparse import ArgumentParser
from condor import condor_booklist, createScript
import json

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser(description="Process input rootfiles into numpy arrays for Hmumu XGBoost analysis.")
    parser.add_argument('-i', '--inputdir', action='store', default='inputs', help='Directory that contains ntuples.')
    return  parser.parse_args()


def arrange_samplelist(channel,category,inputdir):
    samples=[]
    for filename in os.listdir(inputdir):
        if channel not in filename: continue
        if '.root' not in filename: continue
        if os.path.isfile('skimmed_ntuples/%s/%s' % (category, filename)): continue
        if not os.path.isdir('skimmed_ntuples/%s' % category): os.makedirs('skimmed_ntuples/%s' % category)
        samples.append("%s/%s skimmed_ntuples/%s/%s" % (inputdir, filename, category, filename))
    return samples

def main():

    args=getArgs()
    inputdir = args.inputdir

    #CONDA_PREFIX = os.getenv("CONDA_PREFIX").replace("/envs/hmumuml", "")
    CONDA_PREFIX = '/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml' 

    createScript('scripts/submit_skim_ntuples_Hmm.sh', f"""#!/bin/bash

initdir=$1
input=$2
output=$3

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('{CONDA_PREFIX}/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "{CONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
        . "{CONDA_PREFIX}/etc/profile.d/conda.sh"
    else
        export PATH="{CONDA_PREFIX}/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd $initdir
source scripts/setup.sh
echo python scripts/skim_ntuples_Hmm.py -i $input -o $output
python scripts/skim_ntuples_Hmm.py -i $input -o $output
        """)

    with open('data/inputs_config.json') as f:
        config = json.load(f)

    sample_list = config['sample_list']

    condor_list = condor_booklist('scripts/submit_skim_ntuples_Hmm.sh', 'skim_ntuples_Hmm')
    condor_list.initialdir_in_arguments()
    condor_list.set_JobFlavour('workday')

    for category in sample_list:

        for channel in sample_list[category]:

            samples = arrange_samplelist(channel,category,inputdir)
            condor_list.add_Argument(samples)

    condor_list.summary('Basic')
    condor_list.submit()

    
if __name__=='__main__':
    main()

