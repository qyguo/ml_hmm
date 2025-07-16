source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc11-opt/setup.sh

python scripts/train_bdt.py -i /data/pku/home/qyguo/Ntuples/skimmed_ntuples_v4/ -r two_jet --save -f {0,1,2,3} --roc --importance --corr -c  data/training_config_BDT_good.json 
python scripts/apply_bdt.py -i /data/pku/home/qyguo/Ntuples/skimmed_ntuples_v4/ -r two_jet -c data/training_config_BDT_good.json  data/apply_config_BDT.json -m models_0716/
