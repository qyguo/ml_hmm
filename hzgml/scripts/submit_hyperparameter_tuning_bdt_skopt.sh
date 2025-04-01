#!/bin/bash

# initdir=$1
region=$1
fold=$2

# cd $initdir
source scripts/setup.sh
# for region in zero_jet one_jet two_jet VH_ttH VBF; do
region=two_jet
for fold in {0..3};do
echo python scripts/train_bdt.py -r $region -f $fold --skopt --skopt-plot
#python scripts/train_bdt.py -r $region -f $fold --skopt --skopt-plot
done
# done
        
