#!/usr/bin/env python
import copy
import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import uproot
# from root_pandas import *
import pickle
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import xgboost as xgb
from tqdm import tqdm
import logging
from pdb import set_trace
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError + 1
pd.options.mode.chained_assignment = None

import os
from datetime import datetime

# Get the current date in the format MMDD
current_date = datetime.now().strftime("%m%d")

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    #parser.add_argument('-c', '--config', action='store', nargs=2, default=['data/training_config_BDT.json', 'data/apply_config_BDT.json'], help='Region to process')
    parser.add_argument('-c', '--config', action='store', nargs=2, default=['data/training_config_BDT_Hmm_RunIII.json', 'data/apply_config_BDT.json'], help='Region to process')
    parser.add_argument('-i', '--inputFolder', action='store', default='/eos/user/q/qguo/vbfhmm/ml/2018/skimmed_ntuples_v4/', help='directory of training inputs')
    parser.add_argument('-m', '--modelFolder', action='store', default='models', help='directory of BDT models')
    parser.add_argument('-o', '--outputFolder', action='store', default='outputs', help='directory for outputs')
    parser.add_argument('-r', '--region', action='store', choices=['two_jet', 'one_jet', 'zero_jet', 'zero_to_one_jet', 'VH_ttH', 'all_jet'], default='two_jet', help='Region to process')
    parser.add_argument('-cat', '--category', action='store', nargs='+', help='apply only for specific categories')

    parser.add_argument('-s', '--shield', action='store', type=int, default=-1, help='Which variables needs to be shielded')
    parser.add_argument('-a', '--add', action='store', type=int, default=-1, help='Which variables needs to be added')
    parser.add_argument('-F', '--FixSBH125', action='store_true', default=False, help='Fix the H mass to be 125GeV to get the scores')
    parser.add_argument('-y', '--year', action='store', default='', help='directory name')

    return parser.parse_args()

class ApplyXGBHandler(object):
    "Class for applying XGBoost"

    def __init__(self, configPath, region=''):

        print('===============================')
        print('  ApplyXGBHandler initialized')
        print('===============================')

        args=getArgs()
        self._shield = args.shield
        self._add = args.add
        self._FixSBH125 = args.FixSBH125
        self._year = args.year

        self._region = region
        self._inputFolder = ''
        self._inputTree = ''
        #self._inputTree = region if region else 'inclusive'
        #self._inputTree = 'two_jet_m110To150'
        #print("inputTree: ", self._inputTree)
        #self._inputTree = 'two_jet'
        self._modelFolder = ''
        self._outputFolder = ''
        #self._outputFolder = f'models_{current_date}'
        self._chunksize = 500000
        self._category = []
        self._branches = []
        self._outbranches = []

        self.m_models = {}
        self.m_tsfs = {}

        self.train_variables = {}
        self.randomIndex = 'event'

        self.models = {}
        self.observables = []
        self.preselections = []

        self.readApplyConfig(configPath[1])
        self.readTrainConfig(configPath[0])
        self.arrangeBranches()
        self.arrangePreselections()

    def readApplyConfig(self, configPath):
        """Read configuration file formated in json to extract information to fill TemplateMaker variables."""
        try:
            member_variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_") and not attr.startswith('m_')]

            stream = open(configPath, 'r')
            configs = json.loads(stream.read())

            # read from the common settings
            config = configs["common"]
            for member in config.keys():
                if member in member_variables:
                    setattr(self, member, config[member])

            # read from the region specific settings
            if self._region:
                config = configs[self._region]
                for member in config.keys():
                    if member in member_variables:
                        setattr(self, member, config[member])
                if '+preselections' in config.keys():
                    self.preselections += config['+preselections']
                if '+observables' in config.keys():
                    self.observables += config['+observables']

        except Exception as e:
            logging.error("Error reading apply configuration '{config}'".format(config=configPath))
            logging.error(e)

    def readTrainConfig(self, configPath):

        try:
            stream = open(configPath, 'r')
            configs = json.loads(stream.read())
            # if (self._add>=0):
            #     configs["common"]["train_variables"].append(configs["common"]["+train_variables"][self._add])
   
            config = configs["common"]

            if 'randomIndex' in config.keys(): self.randomIndex = config['randomIndex']
            if 'inputTree' in config.keys(): self._inputTree = config['inputTree']
 
            if self.models:
                for model in self.models:
    
                    # read from the common settings
                    config = configs["common"]
                    if 'train_variables' in config.keys(): self.train_variables[model] = config['train_variables'][:]
    
                    # read from the region specific settings
                    if model in configs.keys():
                        config = configs[model]
                        if self._add >= 0:
                            config["+train_variables"].append(config["test_variables"][self._add])
                        if 'train_variables' in config.keys(): self.train_variables[model] = config['train_variables'][:]
                        if '+train_variables' in config.keys(): self.train_variables[model] += config['+train_variables']
                        if self._shield >= 0:
                            self.train_variables[model].pop(self._shield)

                        print("\n\n")
                        print(self.train_variables[model])
                        print(len(self.train_variables[model]))
                        print("\n\n")
            
            if self._region:
                config = configs[self._region]
                if 'inputTree' in config.keys(): self._inputTree = config['inputTree']
                if self._add >= 0:
                    config["+train_variables"].append(config["test_variables"][self._add])
                #for member in config.keys():
                #    if member in member_variables:
                #        setattr(self, member, config[member])
                #if '+train_mc_background' in config.keys():
                #    self.train_mc_background += config['+train_mc_background']
                #if '+train_signal' in config.keys():
                #    self.train_signal += config['+train_signal']
                if '+train_variables' in config.keys():
                    self.train_variables += config['+train_variables']
                #if '+preselections' in config.keys():
                #    self.preselections += config['+preselections']
                #if '+signal_preselections' in config.keys():
                #    self.signal_preselections += config['+signal_preselections']
                #if '+background_preselections' in config.keys():
                #    self.background_preselections += config['+background_preselections']


        except Exception as e:
            logging.error("Error reading training configuration '{config}'".format(config=configPath))
            logging.error(e)

    def arrangeBranches(self):

        self._branches = set()
        for model in self.models:
            self._branches = self._branches | set(self.train_variables[model])

        self._branches = self._branches | set([self.randomIndex]) | set([p.split()[0] for p in self.preselections]) | set(self.observables)
        self._branches = list(self._branches)

        for model in self.models:
            self.train_variables[model] = [x.replace('noexpand:', '') for x in self.train_variables[model]]
        self.preselections = [x.replace('noexpand:', '') for x in self.preselections]
        self.randomIndex = self.randomIndex.replace('noexpand:', '')

        self._outbranches = [branch for branch in self._branches if 'noexpand' not in branch]

    def arrangePreselections(self):

        if self.preselections:
            self.preselections = ['data.' + p for p in self.preselections]

    def setInputFolder(self, inputFolder):
        self._inputFolder = inputFolder

    def setModelFolder(self, modelFolder):
        self._modelFolder = modelFolder

    def setOutputFolder(self, outputFolder):
        self._outputFolder = outputFolder + f'_bdt_{current_date}'
        if self._year:  self._outputFolder = self._outputFolder + '_' + self._year
        if self._FixSBH125:  self._outputFolder += '_SB_HM125'

    def preselect(self, data):

        for p in self.preselections:
            data = data[eval(p)]

        return data

    def loadModels(self):

        if self.models:
            for model in self.models:
                print('XGB INFO: Loading BDT model: ', model)
                self.m_models[model] = []
                for i in range(4):
                    bst = xgb.Booster()
                    bst.load_model('%s/BDT_%s_%d.h5'%(self._modelFolder, model, i))
                    self.m_models[model].append(bst)
                    del bst

    def loadTransformer(self):
        
        if self.models:
            for model in self.models:
                print('XGB INFO: Loading score transformer for model: ', model)
                self.m_tsfs[model] = []
                for i in range(4):
                    tsf = pickle.load(open('%s/BDT_tsf_%s_%d.pkl'%(self._modelFolder, model, i), "rb" ), encoding = 'latin1' )
                    self.m_tsfs[model].append(tsf)

    def applyBDT(self, category, scale=1):
        outputbraches = copy.deepcopy(self._outbranches)
        branches = copy.deepcopy(self._branches)
        # branches += ["Z_sublead_lepton_pt", "gamma_mvaID_WP80", "gamma_mvaID_WPL"]
        branches += ["eventWeight", "trg_single_mu24", "nmuons"]
        outputbraches += ["eventWeight", "trg_single_mu24", "nmuons"]
        # if category == "DYJetsToLL":
        #     branches.append('n_iso_photons')
        # if category != "data_fake" and category != "mc_true" and category != "mc_med":
        #     branches.append('gamma_mvaID_WP80')
        # if category == "data_fake" or category == "mc_true" or category == "mc_med":
        #     branches += ['weight_err']
        #     outputbraches += ['weight_err']
        if category == "mc_true" or category == "mc_med":
            branches += ['tagger']
            outputbraches += ['tagger']

        # print(branches)
        # print(outputbraches)
        
        outputContainer = self._outputFolder + '/' + self._region
        print("outputContainer: ",outputContainer)
        output_path = outputContainer + '/%s.root' % category
        if not os.path.isdir(outputContainer): os.makedirs(outputContainer)
        if os.path.isfile(output_path): os.remove(output_path)

        f_list = []
        cat_folder = self._inputFolder + '/' 
        for f in os.listdir(cat_folder):
            #if f.endswith('{}_ml.root'.format(category)): f_list.append(cat_folder + '/' + f)
            if f.endswith('{}.root'.format(category)): f_list.append(cat_folder + '/' + f)

        print('-------------------------------------------------')
        for f in f_list: print('XGB INFO: Including sample: ', f)

        #TODO put this to the config
        # for data in tqdm(read_root(sorted(f_list), key=self._inputTree, columns=branches, chunksize=self._chunksize), bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}', desc='XGB INFO: Applying BDTs to %s samples' % category):
        with uproot.recreate(output_path) as output_file:
            out_data = pd.DataFrame()
            for filename in tqdm(sorted(f_list), desc='XGB INFO: Applying BDTs to %s samples' % category, bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
                file = uproot.open(filename)
                print(filename, self._inputTree)
                for data in file[self._inputTree].iterate(branches, library='pd', step_size=self._chunksize):
                    data = self.preselect(data)
                    # data = data[data.Z_sublead_lepton_pt >= 15]
                    # if category == "DYJetsToLL":
                    #     data = data[data.n_iso_photons == 0]
                    # if category != "data_fake" and category != "mc_true" and category != "mc_med":
                    #     pass
                    #     # data = data[data.gamma_mvaID_WP80 > 0] #TODO: check this one
                    #     data = data[data.gamma_mvaID_WPL > 0] #TODO: check this one

                    for i in range(4):

                        #data[self.H_mass] = 125
                        #if ( data[data.diMufsr_rc_mass] > 110 & data[data.diMufsr_rc_mass] < 115 ) | ( data[data.diMufsr_rc_mass] > 135 & data[data.diMufsr_rc_mass] < 150 ):
                        #    data[data.diMufsr_rc_mass] = 125
                        if self._FixSBH125:
                            mask = ((data['diMufsr_rc_mass'] > 110) & (data['diMufsr_rc_mass'] < 115)) | ((data['diMufsr_rc_mass'] > 135) & (data['diMufsr_rc_mass'] < 150))
                            data.loc[mask, 'diMufsr_rc_mass'] = 125

                        data_s = data[data[self.randomIndex]%4 == i]
                        data_o = data_s[outputbraches]

                        for model in self.train_variables.keys():
                            x_Events = data_s[self.train_variables[model]]
                            dEvents = xgb.DMatrix(x_Events)
                            scores = self.m_models[model][i].predict(dEvents)
                            if len(scores) > 0:
                                scores_t = self.m_tsfs[model][i].transform(scores.reshape(-1,1)).reshape(-1)
                            else:
                                scores_t = scores
                        
                            xgb_basename = self.models[model]
                            data_o[xgb_basename] = scores
                            data_o[xgb_basename+'_t'] = scores_t

                        out_data = pd.concat([out_data, data_o], ignore_index=True, sort=False)
                #out_data.to_root(output_path, key='test', mode='a', index=False)
                
            # Convert DataFrame to dictionary of arrays
            #out_data_dict = out_data.to_dict('list')

            if not out_data.empty:
                print("not empty")
                # Convert DataFrame to dictionary of arrays
                out_data_dict = out_data.to_dict('list')
                # Write the dictionary of arrays to the ROOT file
                output_file["test"] = out_data_dict
            else:
                print("No data to write to ROOT file.")
            #output_file['test'] = out_data_dict
            del out_data, data_s, data_o


def main():

    args=getArgs()
    
    configPath = args.config
    xgb = ApplyXGBHandler(configPath, args.region)

    xgb.setInputFolder(args.inputFolder)
    xgb.setModelFolder(args.modelFolder)
    xgb.setOutputFolder(args.outputFolder)

    xgb.loadModels()
    xgb.loadTransformer()

    with open('data/inputs_config.json') as f:
        config = json.load(f)
    sample_list = config['sample_list']

    for category in sample_list:
        if args.category and category not in args.category: continue
        xgb.applyBDT(category)

    return

if __name__ == '__main__':
    main()
