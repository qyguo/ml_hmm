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


################
################
#outName_Full={
#   'DYJetsToLL_M-50-madgraphMLM_20UL18_ext1-v1': 'DY_50MLM', 
#   'DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX_RunIIAutumn18': 'DY_105To160', 
#   'DYJetsToLL_M-50-amcatnloFXFX_20UL18': 'DY_50FxFx', 
#   'EWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18': 'EWK_MLL50_MJJ120', 
#   'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18': 'EWK_105To160', 
#   'TTTo2L2Nu-powheg_20UL18': 'TTTo2L2Nu', 
#   'TTToSemiLeptonic-powheg_20UL18': 'TTToSemiLep', 
#   'ST_tW_top_5f_inclusiveDecays-powheg_20UL18': 'ST_tW_top', 
#   'ST_tW_antitop_5f_inclusiveDecays-powheg_20UL18': 'ST_tW_antitop', 
#   'ST_t-channel_top_5f_InclusiveDecays-powheg_20UL18': 'ST_t-channel_top', 
#   'ST_t-channel_antitop_5f_InclusiveDecays-powheg_20UL18': 'ST_t-channel_antitop', 
#   'ST_s-channel_4f_leptonDecays-amcatnlo_20UL18': 'ST_s-channel', 
#   'tZq_ll_4f_ckm_NLO-madgraph_RunIIAutumn18_ext1-v1_20UL18': 'tZq', 
#}
#for key, value in outName_Full.items():
#    print(f'Key: {key}, Value: {value}')
################

################
# Signal samples
################
# type="signal"
# for samples in ggH VBF WplusH WminusH ZH ttH;
# do
# mkdir -p ${target}${samples}
# for year in 2016 2017 2018;
# do
# echo python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_M125_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_M125_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# done
# done

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/ggH/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/VBF/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/WminusH/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/WplusH/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZH/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/ttH/
#python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/ggH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ggH/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/VBFH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/VBF/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/WminusH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/WminusH/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/WplusH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/WplusH/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/ZH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZH/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/signal/ttH_M125_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ttH/2017.root

##############
# Data samples
##############

# type="data"
# for samples in Data;
# do
# mkdir -p ${target}${samples}
# for year in 2016 2017 2018;
# do
# echo python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# done
# done

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/data/
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/data/Data_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/data/2017.root

###################
# Prompt MC samples
###################

# type="bkgmc"
# for samples in ZGToLLG DYJetsToLL WGToLNuG ZG2JToG2L2J EWKZ2J TT TTGJets TGJets ttWJets ttZJets WW WZ ZZ;
# do
# mkdir -p ${target}${samples}
# for year in 2016 2017 2018;
# do
# echo python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i ${input}${type}/${samples}_${year}/merged_nominal.parquet -o ${target}${samples}/${year}.root
# done
# done

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZGToLLG/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZG2JToG2L2J/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/TGJets/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/TTGJets/
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/ZGToLLG_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZGToLLG/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/ZG2JToG2L2J_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZG2JToG2L2J/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/TTGJets_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/TTGJets/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/TGJets_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/TGJets/2017.root

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/DYJetsToLL/
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/dy/DYJetsToLL_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/DYJetsToLL/2017.root

# Use fake photon background estimation with data-driven

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/data_med/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/data_fake/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/mc_true/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/mc_med/
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/apply_weight.py

# ######################
# Non prompt MC samples
# ######################

# mkdir -p /eos/home-j/jiehan/root/2017/skimmed_ntuples/LLAJJ/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/TT/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/WW/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/WZ/ /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZZ/
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/LLAJJ_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/LLAJJ/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/TT_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/TT/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/WW_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/WW/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/WZ_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/WZ/2017.root
# python /afs/cern.ch/user/j/jiehan/private/HiggsZGammaAna/hzgml/scripts/skim_ntuples.py -i /eos/home-j/jiehan/parquet/2017/mva_based/background/ZZ_2017/merged_nominal.parquet -o /eos/home-j/jiehan/root/2017/skimmed_ntuples/ZZ/2017.root

echo "==============FINISHED==========="
