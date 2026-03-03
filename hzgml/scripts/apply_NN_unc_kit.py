#!/usr/bin/env python3
# apply_NN.py
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras as keras

import os
import copy
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm
import logging
import ROOT
from datetime import datetime

ROOT.gErrorIgnoreLevel = ROOT.kError + 1
pd.options.mode.chained_assignment = None

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Get the current date in the format MMDD
current_date = datetime.now().strftime("%m%d")


def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        action="store", nargs=2,
        default=["data/training_config_NN_Hmumu_unc_kit.json", "data/apply_config_NN_unc_kit.json"],
        help="training_config.json apply_config.json"
    )
    parser.add_argument(
        "-i", "--inputFolder",
        action="store",
        default="/eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v4/",
        help="directory of inputs"
    )
    parser.add_argument("-m", "--modelFolder", action="store", default="models", help="directory of models")
    parser.add_argument("-o", "--outputFolder", action="store", default="outputs", help="directory for outputs")
    parser.add_argument(
        "-r", "--region",
        action="store",
        choices=["two_jet", "one_jet", "zero_jet", "VH_ttH"],
        default="two_jet",
        help="Region to process"
    )
    parser.add_argument("-cat", "--category", action="store", nargs="+", help="apply only for specific categories")

    parser.add_argument("-s", "--shield", action="store", type=int, default=-1, help="Which variables needs to be shielded")
    parser.add_argument("-a", "--add", action="store", type=int, default=-1, help="Which variables needs to be added")
    parser.add_argument("-F", "--FixSBH125", action="store_true", default=False, help="Fix H mass to be 125 GeV")
    parser.add_argument("-y", "--year", action="store", default="", help="source year label to write to output")

    return parser.parse_args()


def list_matching_trees(uproot_file, base_tree):
    """Return all TTrees whose name is base_tree or base_tree__*."""
    keys = [k.split(";")[0] for k in uproot_file.keys()]
    trees = []
    for k in keys:
        try:
            obj = uproot_file[k]
        except Exception:
            continue
        if hasattr(obj, "iterate") and (k == base_tree or k.startswith(base_tree + "__")):
            trees.append(k)
    return sorted(trees)


def available_branches(tree, requested):
    """Filter requested branches to those existing in this TTree."""
    try:
        have = set(tree.keys())  # branch names
    except Exception:
        have = set()
    return [b for b in requested if b in have]


class ApplyXGBHandler(object):
    """Class for applying NN models"""

    def __init__(self, configPath, region=""):
        print("===============================")
        print("  ApplyXGBHandler initialized")
        print("===============================")

        args = getArgs()
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year

        self._region = region
        self._inputFolder = ""
        self._modelFolder = ""
        self._outputFolder = ""
        self._chunksize = 500000

        # NOMINAL base tree (systematics are base+"__*")
        self._inputTree = "data_two_jet_m110To150"

        self._category = []
        self._branches = []
        self._outbranches = []

        self.m_models = {}
        self.m_tsfs = {}

        self.train_variables = {}
        self.algorithm = {}
        self.object_variables = {}
        self.other_variables = {}
        self.randomIndex = "event"

        self.models = {}
        self.observables = []
        self.preselections = []

        self.readApplyConfig(configPath[1])
        self.readTrainConfig(configPath[0])
        self.arrangeBranches()
        self.arrangePreselections()

    def readApplyConfig(self, configPath):
        try:
            member_variables = [
                attr for attr in dir(self)
                if not callable(getattr(self, attr)) and not attr.startswith("_") and not attr.startswith("m_")
            ]
            with open(configPath, "r") as stream:
                configs = json.loads(stream.read())

            config = configs["common"]
            for member in config.keys():
                if member in member_variables:
                    setattr(self, member, config[member])

            if self._region:
                config = configs[self._region]
                for member in config.keys():
                    if member in member_variables:
                        setattr(self, member, config[member])
                if "+preselections" in config.keys():
                    self.preselections += config["+preselections"]
                if "+observables" in config.keys():
                    self.observables += config["+observables"]

        except Exception as e:
            logging.error(f"Error reading apply configuration '{configPath}'")
            logging.error(e)

    def readTrainConfig(self, configPath):
        try:
            with open(configPath, "r") as stream:
                configs = json.loads(stream.read())

            config = configs["common"]
            if "randomIndex" in config.keys():
                self.randomIndex = config["randomIndex"]

            if self.models:
                for model in self.models:
                    config = configs["common"]
                    if self._add >= 0:
                        config["train_variables"].append(config["+train_variables"][self._add])

                    if "train_variables" in config.keys():
                        self.train_variables[model] = config["train_variables"][:]
                    if "algorithm" in config.keys():
                        self.algorithm[model] = config["algorithm"]
                    if "object_variables" in config.keys():
                        self.object_variables[model] = config["object_variables"]
                    if "other_variables" in config.keys():
                        self.other_variables[model] = config["other_variables"]

                    if model in configs.keys():
                        config = configs[model]
                        if "train_variables" in config.keys():
                            self.train_variables[model] = config["train_variables"][:]
                        if "+train_variables" in config.keys():
                            self.train_variables[model] += config["+train_variables"]
                        if "algorithm" in config.keys():
                            self.algorithm[model] = config["algorithm"]
                        if "object_variables" in config.keys():
                            self.object_variables[model] = config["object_variables"]
                        if "other_variables" in config.keys():
                            self.other_variables[model] = config["other_variables"]

            if self._shield >= 0 and self.models:
                # If you intended to remove a feature, you probably want to handle this per-model list.
                pass

        except Exception as e:
            logging.error(f"Error reading training configuration '{configPath}'")
            logging.error(e)

    def arrangeBranches(self):
        self._branches = set()
        for model in self.models:
            self._branches |= set(self.train_variables[model])

        # add index/preselections/observables
        self._branches |= set([self.randomIndex])
        self._branches |= set([p.split()[0] for p in self.preselections])
        self._branches |= set(self.observables)

        # Ensure these are read and/or written if present
        self._branches |= set(["eventWeight", "trg_single_mu24", "nmuons"])
        #self._branches |= set(["njets", "jet1_mass", "jet2_mass", "PDF_uncertainty_down", "PDF_uncertainty_up", "qcd_unc_down", "qcd_unc_up", "iso_MuonEffup", "iso_MuonEffdown", "id_MuonEffup", "id_MuonEffdown", "source_year"])
        self._branches |= set(["njets", "jet1_mass", "jet2_mass", "source_year", "cate_index"])

        self._branches = list(self._branches)

        for model in self.models:
            self.train_variables[model] = [x.replace("noexpand:", "") for x in self.train_variables[model]]
        self.preselections = [x.replace("noexpand:", "") for x in self.preselections]
        self.randomIndex = self.randomIndex.replace("noexpand:", "")

        # output branches = everything except "noexpand:*"
        self._outbranches = [branch for branch in self._branches if "noexpand" not in branch]

    def arrangePreselections(self):
        if self.preselections:
            self.preselections = ["data." + p for p in self.preselections]

    def setInputFolder(self, inputFolder):
        self._inputFolder = inputFolder

    def setModelFolder(self, modelFolder):
        self._modelFolder = modelFolder

    def setOutputFolder(self, outputFolder):
        self._outputFolder = outputFolder + f"_dnn_{current_date}"
        if self._year:
            self._outputFolder = self._outputFolder + "_" + self._year
        if self._FixSBH125:
            self._outputFolder += "_SB_HM125"

    def preselect(self, data):
        for p in self.preselections:
            data = data[eval(p)]
        return data

    def loadModels(self):
        if self.models:
            for model in self.models:
                logging.info(f"Loading NN model group: {model}")
                self.m_models[model] = []
                for i in range(1, 5):
                    bst = load_model(f"{self._modelFolder}/merged_model_fold_{i}.h5")
                    self.m_models[model].append(bst)

    def loadTransformer(self):
        import joblib
        if self.models:
            for model in self.models:
                logging.info(f"Loading score transformer for model: {model}")
                self.m_tsfs[model] = []
                for i in range(1, 5):
                    tsf = joblib.load(f"{self._modelFolder}/DNN_tsf_{i}_new.joblib")
                    self.m_tsfs[model].append(tsf)

    def loadScaler(self):
        import joblib
        self.scaler_l = joblib.load(f"{self._modelFolder}/scaler_new.pkl")

    def applyBDT(self, category, scale=1):

        NOMINAL_ONLY_UNCS = [
            "PDF_uncertainty_down",
            "PDF_uncertainty_up",
            "qcd_unc_down",
            "qcd_unc_up",
            "iso_MuonEffup",
            "iso_MuonEffdown",
            "id_MuonEffup",
            "id_MuonEffdown",
        ]

        branches = copy.deepcopy(self._branches)
        outbranches = copy.deepcopy(self._outbranches)

        def format_twojet_inputs(x, model):
            x_mass = x[:, self.object_variables[model]]
            x_others = x[:, self.other_variables[model]]
            return [x_mass, x_others]

        outputContainer = f"{self._outputFolder}/{self._region}"
        output_path = f"{outputContainer}/{category}.root"
        os.makedirs(outputContainer, exist_ok=True)
        if os.path.isfile(output_path):
            os.remove(output_path)

        # Input ROOT files for this category
        f_list = []
        cat_folder = self._inputFolder.rstrip("/") + "/"
        for f in os.listdir(cat_folder):
            if f.endswith(f"{category}.root"):
                f_list.append(cat_folder + f)

        if len(f_list) == 0:
            logging.warning(f"No input files found for category={category} in {cat_folder}")
            return

        base_tree = self._inputTree

        with uproot.recreate(output_path) as output_file:
            created_trees = set()

            for filename in tqdm(sorted(f_list),
                                 desc=f"Applying NN to {category}",
                                 bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}"):

                fin = uproot.open(filename)
                tree_names = list_matching_trees(fin, base_tree)
                if not tree_names:
                    logging.warning(f"No matching trees in {filename} for base {base_tree}")
                    continue

                for tree_name in tree_names:
                    #tree = fin[tree_name]
                    #use_branches = available_branches(tree, branches)
                    tree = fin[tree_name]

                    # Nominal vs systematic tree
                    is_nominal_tree = (tree_name == base_tree)
                    
                    # Only read uncertainty branches for the nominal tree
                    extra_unc_branches = NOMINAL_ONLY_UNCS if is_nominal_tree else []
                    
                    use_branches = available_branches(
                        tree,
                        branches + extra_unc_branches
                    )

                    #if is_nominal_tree:
                    #    logging.info(f"[NN] Writing nominal-only uncertainties for {tree_name}")

                    # Iterate chunks
                    for data in tree.iterate(use_branches, library="pd", step_size=self._chunksize):
                        # Ensure output-required columns exist even if branch missing in this tree
                        for col in ["njets", "jet1_mass", "jet2_mass", "cate_index"]:
                            if col not in data.columns:
                                data[col] = -999.0

                        data = self.preselect(data)
                        if data.empty:
                            continue

                        # Fold split
                        for i in range(4):
                            if self._FixSBH125 and ("diMufsr_kit_BSC_mass" in data.columns):
                                mask = ((data["diMufsr_kit_BSC_mass"] > 110) & (data["diMufsr_kit_BSC_mass"] < 115)) | \
                                       ((data["diMufsr_kit_BSC_mass"] > 135) & (data["diMufsr_kit_BSC_mass"] < 150))
                                data.loc[mask, "diMufsr_kit_BSC_mass"] = 125

                            if self.randomIndex not in data.columns:
                                logging.error(f"randomIndex '{self.randomIndex}' missing in tree {tree_name}")
                                continue

                            data_s = data[data[self.randomIndex] % 4 == i]
                            if data_s.empty:
                                continue

                            ## Only keep output branches that exist, plus our required extras
                            #keep_cols = [c for c in outbranches if c in data_s.columns]
                            #for c in ["njets", "jet1_mass", "jet2_mass"]:
                            #    if c not in keep_cols:
                            #        keep_cols.append(c)

                            #data_o = data_s[keep_cols].copy()

                            keep_cols = [c for c in outbranches if c in data_s.columns]

                            # Always keep jet info
                            for c in ["njets", "jet1_mass", "jet2_mass"]:
                                if c not in keep_cols:
                                    keep_cols.append(c)
                            
                            # Only keep uncertainty branches for NOMINAL tree
                            if is_nominal_tree:
                                for c in NOMINAL_ONLY_UNCS:
                                    if c in data_s.columns:
                                        keep_cols.append(c)
                            
                            data_o = data_s[keep_cols].copy()

                            # Add source_year to output
                            #data_o["source_year"] = int(self._year) if self._year else -1

                            for model in self.train_variables.keys():
                                # ensure all needed training vars exist
                                tv = self.train_variables[model]
                                missing = [x for x in tv if x not in data_s.columns]
                                if missing:
                                    logging.error(f"Missing train vars in {tree_name}: {missing[:5]}{'...' if len(missing)>5 else ''}")
                                    continue

                                x_Events = data_s[tv].to_numpy()
                                x_Events = self.scaler_l.transform(x_Events)

                                if self.algorithm[model] in ["RNNGRU", "DeepSets", "SelfAttention", "NN"]:
                                    x_Events = format_twojet_inputs(x_Events, model)

                                scores = self.m_models[model][i].predict(x_Events)
                                if len(scores) > 0:
                                    scores_t = self.m_tsfs[model][i].transform(scores.reshape(-1, 1)).reshape(-1)
                                else:
                                    scores_t = scores

                                NN_basename = self.models[model]
                                data_o[NN_basename] = scores
                                data_o[NN_basename + "_t"] = scores_t
                                data_o[NN_basename + "_arctanh"] = np.arctanh(scores)
                                data_o[NN_basename + "_t_arctanh"] = np.arctanh(scores_t)

                            # Write to output tree with SAME name as input tree
                            out_tree_name = tree_name
                            out_dict = {k: np.asarray(v) for k, v in data_o.to_dict("list").items()}

                            if out_tree_name not in created_trees:
                                output_file.mktree(out_tree_name, {k: out_dict[k].dtype for k in out_dict.keys()})
                                created_trees.add(out_tree_name)

                            output_file[out_tree_name].extend(out_dict)

        logging.info(f"Wrote output: {output_path}")


def main():
    args = getArgs()

    configPath = args.config
    xgb = ApplyXGBHandler(configPath, args.region)

    xgb.setInputFolder(args.inputFolder)
    xgb.setModelFolder(args.modelFolder)
    xgb.setOutputFolder(args.outputFolder)

    xgb.loadModels()
    xgb.loadTransformer()
    xgb.loadScaler()

    # Your sample list config
    with open("data/inputs_config_24_ggH_unc.json") as f:
        config = json.load(f)
    sample_list = config["sample_list"]

    for category in sample_list:
        if args.category and category not in args.category:
            continue
        xgb.applyBDT(category)

    return


if __name__ == "__main__":
    main()
