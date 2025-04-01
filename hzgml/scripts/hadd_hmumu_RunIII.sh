#!/bin/bash                                                                                                                                                                       
echo "==============STARTED=============="

#input="/eos/user/q/qguo/vbfhmm/UL18_check/After_hadd/"
#input="/eos/user/q/qguo/vbfhmm/2022EE_Inc_v2/After_hadd/"
#target="/eos/user/q/qguo/vbfhmm/ml/2022EE/skimmed_ntuples/"
#version of all not just vbf
input="/eos/user/q/qguo/vbfhmm/2023_v4_jetVeto/After_hadd/"
target="/eos/user/q/qguo/vbfhmm/ml/2023_v4_jetVeto/skimmed_ntuples/"

script_Path='/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/scripts/'
# Check if the directory exists, if not create it
if [ ! -d "$target" ]; then
    mkdir -p "$target"
    echo "Directory $target created."
else
    echo "Directory $target already exists."
fi

# Define an associative array
declare -A outName_Full=(
    ["DYJetsToLL_M-50-madgraphMLM_2023"]="DY_50MLM"
    ["DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX_2023"]="DY_105To160"
    ["DYto2L-2Jets_MLL-50_amcatnloFXFX_2023"]="DY_50FxFx"
    #["EWK_LLJJ_MLL-50_MJJ-120-madgraph"]="EWK_LLJJ_M50"
    #["EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected"]="EWK_LLJJ_M105To160"
    ["TTto2L2Nu_powheg_2023"]="TTTo2L2Nu"
    #["TTToSemiLeptonic-powheg_20UL18"]="TTToSemiLep"
    ["TWminusto2L2Nu_powheg_2023"]="ST_tW_top"
    ["TbarWplusto2L2Nu_powheg_2023"]="ST_tW_antitop"
    #["ST_t-channel_top_5f_InclusiveDecays-powheg_20UL18"]="ST_t-channel_top"
    #["ST_t-channel_antitop_5f_InclusiveDecays-powheg_20UL18"]="ST_t-channel_antitop"
    #["ST_s-channel_4f_leptonDecays-amcatnlo_20UL18"]="ST_s-channel"
    #["TZQB-4FS_OnshellZ_amcatnlo_2023"]="tZq"
    ["GluGluHto2Mu_M-125_powheg_2023"]="GluGluHToMuMu_M125"
    ["VBFHto2Mu_M-125_powheg_2023"]="VBFHToMuMu_M125"
    ["Muon01_23BC_v1234"]="data"
    #["DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM_RunIIAutumn18"]="DY_105To160_VBFFilter"
    ["ZZto4L_powheg_2023"]="ZZTo4L"
    ["ZZto2L2Nu_powheg_2023"]="ZZTo2L2Nu"
    ["WWto2L2Nu_powheg_2023"]="WWTo2L2Nu"
    ["WZto3LNu_powheg_2023_ext1-v2"]="WZTo3LNu"
    ["ZZto2L2Q_powheg_2023"]="ZZTo2L2Q"
    ["WZto2L2Q_powheg_2023"]="WZTo2L2Q"
)

# Loop through each key-value pair in the associative array
for key in "${!outName_Full[@]}"; do
    value=${outName_Full[$key]}
    echo "Key: $key, Value: $value"
    echo python ${script_Path}/skim_ntuples_Hmm_23.py -i ${input}/${key}.root -o ${target}/${value}.root -y 2023 >> skim_23_v4_jetVeto.sh 
done


# done

echo "==============FINISHED==========="
