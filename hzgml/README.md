# Welcome to HmumuML

HmumuML is a package that can be used to study the XGBoost BDT categorization in Hmumu analysis.

## Setup

This project requires to use python3 and either conda or virtual environment. If you want to use a conda environment, install Anaconda or Miniconda before setting up.

### First time setup on lxplus

First checkout the code:

```
[outdated] git clone git@github.com:chenzhou36/hmumuml.git [-b your_branch]
cd hzgml
```

Then,

```
source scripts/install.sh
```

### Normal setup on lxplus

After setting up the environment for the first time, you can return to this setup by doing `source scripts/setup.sh`

## Scripts to run the tasks

### Prepare the training inputs (skimmed ntuples)

The core script is `skim_ntuples.py`. The script will apply the given skimming cuts to input samples and produce corresponding skimmed ntuples. You can run it locally for any specified input files. In H->mumu, it is more convenient and time-efficient to use `submit_skim_ntuples.py` which will find all of the input files specified in `data/inputs_config.json` and run the core script by submitting the condor jobs.

- The script is hard-coded currently, meaning one needs to directly modify the core script to change the variable calculation and the skimming cuts.
- The output files (skimmed ntuples) will be saved to the folder named `skimmed_ntuples` by default.
- `submit_skim_ntuples.py` will only submit the jobs for those files that don't exist in the output folder.
- `run_skim_ntuples.sh` will run all skim jobs locally

```
sh scripts/run_skim_ntuples.sh
```
or
```
python scripts/skim_ntuples.py [-i input_file_path] [-o output_file_path]
```

### Start XGBoost analysis!

The whole ML task consists of training, applying the weights, optimizing the BDT boundaries for categorization, and calculating the number counting significances. The wrapper script `runbdt_all.sh` will run everything. Please have a look!

#### Make some directories
```
mkdir -p models outputs plots
mkdir -p models_$(date +%m%d) outputs_$(date +%m%d) plots_$(date +%m%d)
```

#### Training a model

If the hyperparameters are needed to be tuned, please run the following code.
```
source scripts/submit_hyperparameter_tuning_bdt_skopt.sh
```
This is an example for `two_jet` channel. You can change the `region` and `fold` in the script.

The training script `train_bdt.py` will train the model in four-fold, and transform the output scores such that the unweighted signal distribution is flat. The detailed settings, including the preselections, training variables, hyperparameters, etc, are specified in the config file `data/training_config.json`.

```
python scripts/train_bdt.py [-r TRAINED_MODEL] [-f FOLD] [--save]

Usage:
  -r, --region        The model to be trained. Choices: 'zero_jet', 'one_jet', 'two_jet' or 'VBF'.
  -f, --fold          Which fold to run. Default is -1 (run all folds)
  --save              To save the model into HDF5 files, and the pickle files
```

python scripts/train_bdt.py -i /eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v3/ -r two_jet --save -f {0,1,2,3} --corr --skopt --skopt-plot
#python scripts/train_bdt.py -i /eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v3/ -r two_jet --save -f 0 --corr --importance --roc  --skopt --skopt-plot 
python scripts/train_bdt.py -i /eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v3/ -r two_jet --save -f {0,1,2,3} --importance --roc
python scripts/apply_bdt.py -m /afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/models_0617/ -r two_jet 
python scripts/apply_NN.py -m /eos/user/q/qguo/SWAN_projects/ML_test/saved_modela_0616_v2/ -r two_jet -F True
python scripts/categorization_1D_Hmumu.py -r two_jet -b 10 -i outputs_dnn_0624 -t 


###19 variable without the qgl-Likelihood discriminant
### -y is used to add name tag into the outputname
python scripts/apply_NN.py -m /eos/user/q/qguo/SWAN_projects/ML_test/saved_modela_0729_v2/ -r two_jet  -y 2022EE   -i /eos/user/q/qguo/vbfhmm/ml/2022EE/skimmed_ntuples/ -F
python scripts/apply_bdt.py -m models_0729/   -r two_jet  -y 2022EE   -i /eos/user/q/qguo/vbfhmm/ml/2022EE/skimmed_ntuples/ -F

#### Applying the weights

Applying the trained model (as well as the score transformation) to the skimmed ntuples to get BDT scores for each event can be done by doing:
```
python scripts/apply_bdt.py [-r REGION]
```
The script will take the settings specified in the training config file `data/training_config.json` and the applying config file `data/apply_config.json`.

#### Optimizing the BDT boundaries

`categorization_1D.py` will take the Higgs classifier scores of the samples and optimize the boundaries that give the best combined significance. `categorization_2D.py`, on the other hand, takes both the Higgs classifier scores and the VBF classifier scores of the samples and optimizes the 2D boundaries that give the best combined significance.

```
python scripts/categorization_1D.py [-r REGION] [-f NUMBER OF FOLDS] [-b NUMBER OF CATEGORIES] [-n NSCAN] [--floatB] [--minN minN] [--skip]

Usage:
  -f, --fold          Number of folds of the categorization optimization. Default is 1.
  -b, --nbin          Number of BDT categories
  -n, --nscan         Number of scans. Default is 100
  --minN,             minN is the minimum number of events required in the mass window. The default is 5.
  --floatB            To float the last BDT boundary, which means to veto the lowest BDT score events
  --skip              To skip the hadd step (if you have already merged signal and background samples)
```

```
python scripts/categorization_2D.py [-r REGION] [-f NUMBER OF FOLDS] [-b NUMBER OF CATEGORIES] [-b NUMBER OF ggF CATEGORIES] [-n NSCAN] [--floatB] [--minN minN] [--skip]

Usage:
  -f, --fold          Number of folds of the categorization optimization. Default is 1.
  -b, --nbin          Number of BDT categories
  -n, --nscan         Number of scans. Default is 100
  --minN,             minN is the minimum number of events required in the mass window. The default is 5.
  --floatB            To float the last BDT boundary, which means to veto the lowest BDT score events
  --skip              To skip the hadd step (if you have already merged signal and background samples)
```
