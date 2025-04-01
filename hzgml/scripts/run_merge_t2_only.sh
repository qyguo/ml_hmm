#!/bin/bash

dirs=(/eos/user/q/qguo/vbfhmm/ml/2022_v4_jetVeto/skimmed_ntuples/ /eos/user/q/qguo/vbfhmm/ml/2022EE_v4_jetVeto/skimmed_ntuples/ /eos/user/q/qguo/vbfhmm/ml/2023_v4_jetVeto/skimmed_ntuples/ /eos/user/q/qguo/vbfhmm/ml/2023BPix_v4_jetVeto/skimmed_ntuples/)
output_dir=/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples

# Loop over filenames (assuming all in /AA)
for f in /eos/user/q/qguo/vbfhmm/ml/2023_v4_jetVeto/skimmed_ntuples/*.root; do
  base=$(basename "$f")  # A.root, B.root, etc.

  # Build input list
  inputs=()
  for d in "${dirs[@]}"; do
    inputs+=("$d/$base")
  done

  #echo "Merging ${inputs[*]} -> $output_dir/$base"
  echo ""
  #echo "python merge_t2_only.py \"$output_dir/$base\" \"${inputs[@]}\""
  echo "hadd" $output_dir/$base ${inputs[@]}
  #python merge_t2_only.py "$output_dir/$base" "${inputs[@]}"
done

