#!/usr/bin/env python
#
# Created QianyingGUO 05.22.2024
#
import os
import math
from argparse import ArgumentParser
from ROOT import Math, TVector2, TVector3, TLorentzVector
import numpy as np
#import time
import pandas as pd
import uproot
# from root_pandas import *
from tqdm import tqdm
#from scripts.ZSelector import *
from scripts.ZSelector_Horn import *
import awkward as ak
from out_dict import *
#from Datasets.Hmumu.Data_22 import *
#from Datasets.Hmumu.HToMumu_22 import *
#from Datasets.Hmumu.Background_22 import *
from Datasets.Hmumu.Data_22EE import *
from Datasets.Hmumu.HToMumu_22EE import *
from Datasets.Hmumu.Background_22EE import *

def getArgs():
    parser = ArgumentParser(description="Skim the input ntuples for Hmumu XGBoost analysis.")
    parser.add_argument('-i', '--input', action='store', default='inputs', help='Path to the input ntuple')
    parser.add_argument('-o', '--output', action='store', default='outputs', help='Path to the output ntuple')
    parser.add_argument('-y', '--year', action='store', default='2022EE', help='which year of DataSet')
    parser.add_argument('--chunksize', type=int, default=500000, help='size to process at a time') 
    return  parser.parse_args()

def preselect(data):
    #data.query('(Muons_Minv_MuMu_Paper >= 110) | (Event_Paper_Category >= 17)', inplace=True)
    #data.query('Event_Paper_Category > 0', inplace=True)
    #data.query('ngenjets <= 0', inplace=True)

    mask = data["ngenjets"] <= 1
    filtered_data = {key: value[mask] for key, value in data.items()}
    return filtered_data
    #return data

def decorate_hmm(data):

    if data.shape[0] == 0: return data
    data['muon_mass']=0.1056584

    #data['HZ_relM'] = data.H_mass / data.Z_mass
    #data['H_relpt'] = data.H_pt / data.H_mass
    #data['Z_relpt'] = data.Z_pt / data.H_mass
    #data['Z_lead_lepton_relpt'] = data.Z_lead_lepton_pt / data.H_mass
    #data['Z_sublead_lepton_relpt'] = data.Z_sublead_lepton_pt / data.H_mass
    #data['gamma_relpt'] = data.gamma_pt / data.H_mass
    #data['jet_1_relpt'] = data.jet_1_pt / data.H_mass
    #data['jet_2_relpt'] = data.jet_2_pt / data.H_mass
    #data['MET_relpt'] = data.MET_pt / data.H_mass
    #data['gamma_ptRelErr'] = data.apply(lambda x:compute_gamma_relEerror(x), axis=1)
    #data['G_ECM'] = data.apply(lambda x:compute_G_ECM(x), axis=1)
    #data['Z_ECM'] = data.apply(lambda x:compute_Z_ECM(x), axis=1)
    #data['Z_rapCM'] = data.apply(lambda x:compute_Z_rapCM(x), axis=1)
    #data['l_rapCM'] = data.apply(lambda x:compute_l_rapCM(x), axis=1)
    #data['HZ_deltaRap'] = data.apply(lambda x:compute_HZ_deltaRap(x), axis=1)
    #data['l_cosProdAngle'] = data.apply(lambda x:compute_l_prodAngle(x), axis=1)
    #data['Z_cosProdAngle'] = data.apply(lambda x:compute_Z_prodAngle(x), axis=1)
    #data['ll_deltaR'] = data.apply(lambda x:compute_ll_deltaR(x), axis=1)
    #data['leadLG_deltaR'] = data.apply(lambda x:compute_leadLG_deltaR(x), axis=1)
    #data['ZG_deltaR'] = data.apply(lambda x:compute_ZG_deltaR(x), axis=1)
    #data['subleadLG_deltaR'] = data.apply(lambda x:compute_subleadLG_deltaR(x), axis=1)
    #data['H_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'H_phi'), axis=1)
    #data['Z_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_phi'), axis=1)
    #data['Z_lead_lepton_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_lead_lepton_phi'), axis=1)
    #data['Z_sublead_lepton_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_sublead_lepton_phi'), axis=1)
    #for i in np.arange(1,5):
    #    data['jet_%d_deltaphi' %i] = data.apply(lambda x: compute_Delta_Phi(x, "jet", min_jet=i), axis=1)
    #    data['jet%dG_deltaR' %i] = data.apply(lambda x: compute_Delta_R(x, min_jet=i), axis=1)
    #data['additional_lepton_1_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'additional_lepton_1_phi', min_jet=0), axis=1)
    #data['additional_lepton_2_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'additional_lepton_2_phi', min_jet=0), axis=1) 
    #data['MET_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'MET_phi'), axis=1)
    #data['weight'] = data.weight_central
    #data['mass_jj'] = data.apply(lambda x: compute_mass_jj(x), axis=1)
    #data['H_ptt'] = data.apply(lambda x: compute_H_ptt(x), axis=1)
    #data['H_al'] = data.apply(lambda x: compute_H_al(x), axis=1)
    #data['H_bt'] = data.apply(lambda x: compute_H_bt(x), axis=1)
    #data['Z_cos_theta'] = data.apply(lambda x:compute_Z_cosTheta(x), axis=1)
    #data['lep_cos_theta'] = data.apply(lambda x: compute_l_costheta(x), axis=1)
    #data['lep_phi'] = data.apply(lambda x: compute_l_phi(x), axis=1)
    #data['l1g_deltaR'] = data.apply(lambda x: compute_dR1lg(x), axis=1) 
    #data['l2g_deltaR'] = data.apply(lambda x: compute_dR2lg(x), axis=1)
    #data['delta_eta_jj'] = data.apply(lambda x: compute_delta_eta_jj(x), axis=1)
    #data['delta_phi_jj'] = data.apply(lambda x: compute_delta_phi_jj(x), axis=1)
    #data['delta_phi_zgjj'] = data.apply(lambda x: compute_delta_phi_zg_jj(x), axis=1)
    data['H_zeppenfeld'] = data.apply(lambda x: compute_H_zeppenfeld(x), axis=1)
    data['z_star_H_zeppenfeld'] = data.apply(lambda x: compute_z_star_H_zeppenfeld(x), axis=1)
    data['R_pt'] = data.apply(lambda x: compute_pt_balance(x), axis=1)
    #data['pt_balance_0j'] = data.apply(lambda x: compute_pt_balance_0j(x), axis=1)
    #data['pt_balance_1j'] = data.apply(lambda x: compute_pt_balance_1j(x), axis=1)
    #data[['Jets_QGscore_Lead', 'Jets_QGflag_Lead', 'Jets_QGscore_Sub', 'Jets_QGflag_Sub']] = data.apply(lambda x: compute_QG(x), axis=1, result_type='expand')

    #data.rename(columns={'Muons_Minv_MuMu_Paper': 'm_mumu', 'Muons_Minv_MuMu_VH': 'm_mumu_VH', 'EventInfo_EventNumber': 'eventNumber', 'Jets_jetMultip': 'n_j'}, inplace=True)
    #data.drop(['PassesttHSelection', 'PassesVHSelection', 'GlobalWeight', 'SampleOverlapWeight', 'EventWeight_MCCleaning_5'], axis=1, inplace=True)
    data = data.astype(float)
    data = data.astype({'njets': int, 'nelectrons': int, 'nmuons': int, 'event': int})

    return data
###

def decorate_vbfhmm(data):

    if data.shape[0] == 0: return data
    return data

def getLumi(year):
    if year == "2018":  lumi = 59.8*1000
    elif year == "2017":  lumi = 41.4*1000
    elif year == "2016":  lumi = 35*1000
    elif year == "2022":  lumi = (5.0707+3.0063)*1000
    elif year == "2022EE":  lumi = (5.8783+18.007+3.1219)*1000
    elif year == "2023":  lumi = 17.794*1000
    elif year == "2023BPix":  lumi = 9.451*1000
    return lumi

def getShortName(FUllName):
    outName_Full={
       'DYJetsToLL_M-50-madgraphMLM_2022EE': 'DY_50MLM',
       'DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX_2022EE': 'DY_105To160',
       'DYto2L-2Jets_MLL-50_amcatnloFXFX_2022EE_ext1-v1': 'DY_50FxFx',
       #'EWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18': 'EWK_LLJJ_M50',
       #'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18': 'EWK_LLJJ_M105To160',
       'TTto2L2Nu_powheg_2022EE_ext1-v2': 'TTTo2L2Nu',
       #'TTToSemiLeptonic-powheg_20UL18': 'TTToSemiLep',
       'TWminusto2L2Nu_powheg_2022EE_ext1-v2': 'ST_tW_top',
       'TbarWplusto2L2Nu_powheg_2022EE_ext1-v2': 'ST_tW_antitop',
       #'ST_t-channel_top_5f_InclusiveDecays-powheg_20UL18': 'ST_t-channel_top',
       #'ST_t-channel_antitop_5f_InclusiveDecays-powheg_20UL18': 'ST_t-channel_antitop',
       #'ST_s-channel_4f_leptonDecays-amcatnlo_20UL18': 'ST_s-channel',
       'TZQB-4FS_OnshellZ_amcatnlo_2022EE': 'tZq',
       #'GluGluHto2Mu_M-125_powheg_2022EE': 'GluGluHToMuMu_M125',
       #'VBFHto2Mu_M-125_powheg_2022EE': 'VBFHToMuMu_M125',
       'GluGluHto2Mu_M-125_powheg_2022EE': 'GluGluHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_2022EE': 'VBFHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_herwig_2022EE': 'VBFHto2Mu_M-125_herwig',
       'Muon_UL22EE': 'data',
       #'DYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM_RunIIAutumn18': 'DY_105To160_VBFFilter',
       'ZZto4L_powheg_2022EE_ext1-v2': 'ZZTo4L',
       'ZZto2L2Nu_powheg_2022EE_ext1-v2': 'ZZTo2L2Nu',
       'WWto2L2Nu_powheg_2022EE_ext1-v2': 'WWTo2L2Nu',
       'WZto3LNu_powheg_2022EE': 'WZTo3LNu',
       'ZZto2L2Q_powheg_2022EE_ext1-v2': 'ZZTo2L2Q',
       'WZto2L2Q_powheg_2022EE_ext1-v2': 'WZTo2L2Q',
       # 22
       'DYJetsToLL_M-50-madgraphMLM_2022': 'DY_50MLM',
       'DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX_2022': 'DY_105To160',
       'DYto2L-2Jets_MLL-50_amcatnloFXFX_2022_ext1-v1': 'DY_50FxFx',
       #'EWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18': 'EWK_LLJJ_M50',
       #'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18': 'EWK_LLJJ_M105To160',
       'TTto2L2Nu_powheg_2022_ext1-v2': 'TTTo2L2Nu',
       #'TTToSemiLeptonic-powheg_20UL18': 'TTToSemiLep',
       'TWminusto2L2Nu_powheg_2022_ext1-v2': 'ST_tW_top',
       'TbarWplusto2L2Nu_powheg_2022_ext1-v2': 'ST_tW_antitop',
       'TZQB-4FS_OnshellZ_amcatnlo_2022': 'tZq',
       'GluGluHto2Mu_M-125_powheg_2022': 'GluGluHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_2022': 'VBFHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_herwig_2022': 'VBFHto2Mu_M-125_herwig',
       'Muon_UL22': 'data',
       'ZZto4L_powheg_2022_ext1-v2': 'ZZTo4L',
       'ZZto2L2Nu_powheg_2022_ext1-v2': 'ZZTo2L2Nu',
       'WWto2L2Nu_powheg_2022_ext1-v2': 'WWTo2L2Nu',
       'WZto3LNu_powheg_2022': 'WZTo3LNu',
       'ZZto2L2Q_powheg_2022_ext1-v2': 'ZZTo2L2Q',
       'WZto2L2Q_powheg_2022_ext1-v2': 'WZTo2L2Q',
       #2023
       'DYJetsToLL_M-50-madgraphMLM_2023': 'DY_50MLM',
       'DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX_2023': 'DY_105To160',
       'DYto2L-2Jets_MLL-50_amcatnloFXFX_2023': 'DY_50FxFx',
       #'EWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18': 'EWK_LLJJ_M50',
       #'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18': 'EWK_LLJJ_M105To160',
       'TTto2L2Nu_powheg_2023': 'TTTo2L2Nu',
       #'TTToSemiLeptonic-powheg_20UL18': 'TTToSemiLep',
       'TWminusto2L2Nu_powheg_2023': 'ST_tW_top',
       'TbarWplusto2L2Nu_powheg_2023': 'ST_tW_antitop',
       'TZQB-4FS_OnshellZ_amcatnlo_2023': 'tZq',
       'GluGluHto2Mu_M-125_powheg_2023': 'GluGluHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_2023': 'VBFHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_herwig_2023': 'VBFHto2Mu_M-125_herwig',
       'Muon01_23BC_v1234': 'data',
       'ZZto4L_powheg_2023': 'ZZTo4L',
       'ZZto2L2Nu_powheg_2023': 'ZZTo2L2Nu',
       'WWto2L2Nu_powheg_2023': 'WWTo2L2Nu',
       'WZto3LNu_powheg_2023_ext1-v2': 'WZTo3LNu',
       'ZZto2L2Q_powheg_2023': 'ZZTo2L2Q',
       'WZto2L2Q_powheg_2023': 'WZTo2L2Q',
       #2023BPix
       'DYJetsToLL_M-50-madgraphMLM_2023BPix': 'DY_50MLM',
       'DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX_2023BPix': 'DY_105To160',
       'DYto2L-2Jets_MLL-50_amcatnloFXFX_2023BPix': 'DY_50FxFx',
       #'EWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18': 'EWK_LLJJ_M50',
       #'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18': 'EWK_LLJJ_M105To160',
       'TTto2L2Nu_powheg_2023BPix': 'TTTo2L2Nu',
       #'TTToSemiLeptonic-powheg_20UL18': 'TTToSemiLep',
       'TWminusto2L2Nu_powheg_2023BPix': 'ST_tW_top',
       'TbarWplusto2L2Nu_powheg_2023BPix': 'ST_tW_antitop',
       'TZQB-4FS_OnshellZ_amcatnlo_2023BPix': 'tZq',
       'GluGluHto2Mu_M-125_powheg_2023BPix': 'GluGluHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_2023BPix': 'VBFHto2Mu_M-125',
       'VBFHto2Mu_M-125_powheg_herwig_2023BPix': 'VBFHto2Mu_M-125_herwig',
       'Muon01_23D_v12': 'data',
       'ZZto4L_powheg_2023BPix': 'ZZTo4L',
       'ZZto2L2Nu_powheg_2023BPix': 'ZZTo2L2Nu',
       'WWto2L2Nu_powheg_2023BPix': 'WWTo2L2Nu',
       'WZto3LNu_powheg_2023BPix_ext1-v2': 'WZTo3LNu',
       'ZZto2L2Q_powheg_2023BPix': 'ZZTo2L2Q',
       'WZto2L2Q_powheg_2023BPix': 'WZTo2L2Q',
    }
    ShortName = outName_Full[FUllName]
    #for key, value in outName_Full.items():
    #    print(f'Key: {key}, Value: {value}')
    ################
    return ShortName


def combXS(xs_sig,xs_bkg):
        xs = {}
        for s in xs_sig:
            xs[s] = xs_sig[s]
        for s in xs_bkg:
            xs[s] = xs_bkg[s]

        xs["data"] = 1
        xs["fake"] = 1

        return xs

def main():
    
    args = getArgs()

    #variables = [
    #    'H_pt', 'H_eta', 'H_phi', 'H_mass',
    #    'dijet_eta','dijet_mass',
    #    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', #'jet1_btagDeepFlavB',
    #    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', #'jet2_btagDeepFlavB',
    #    'njets', 'nelectrons', 'nmuons', #'n_iso_photons',
    #    'nbjets_loose','nbjets_medium',
    #    'met_pt', 'met_phi',
    #    'mu1_fromH_eta', 'mu1_fromH_phi', 'mu1_fromH_pt',
    #    'mu2_fromH_eta', 'mu2_fromH_phi', 'mu2_fromH_pt', 
    #    'mu1_mu2_dphi', 'mumuH_dR',
    #    'mu1_fromH_ptErr', 'mu2_fromH_ptErr',
    #    'pt_rc_1', 'pt_rc_2',
    #    'fsrPhoton_pt_1', 'fsrPhoton_eta_1', 'fsrPhoton_phi_1', 'fsrPhoton_dROverEt2_1', 'fsrPhoton_relIso03_1',
    #    'fsrPhoton_pt_2', 'fsrPhoton_eta_2', 'fsrPhoton_phi_2', 'fsrPhoton_dROverEt2_2', 'fsrPhoton_relIso03_2',
    #    'trg_single_mu24',
    #    'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2',
    #    'puweight', 'genWeight',
    #    'event'
    #]

    print("Now!!! Porcessing {:s} ......".format(args.input))
    lumi=getLumi(args.year)
    #ShortName=(args.output).split('/')[-1].split('.root')[0]
    FullName=(args.input).split('/')[-1].split('.root')[0]
    ShortName=getShortName(FullName)

    if os.path.isfile(args.output): os.remove(args.output)

    initial_events = 0
    final_events = 0
    isSignal = 0
    isMC = 1
    if "HToMuMu" in args.input or "Hto2Mu" in args.input:
        isSignal = 1
    if "Muon" in args.input:
        isMC = 0
    if not isMC:  variables = data_vars
    elif isSignal:  variables = signal_vars
    else: variables = background_vars
    #else: variables = background_vars + ["ngenjets"]

#    for data in tqdm(read_root(args.input, key='DiMuonNtuple', columns=variables, chunksize=args.chunksize), desc='Processing %s' % args.input, bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):

    #data = pd.read_parquet(args.input)
    print("Reading in ",end='',flush=True)
    file = uproot.open(args.input)
    print("Is signal: ", isSignal)
    print("Is MC: ", isMC)

    GENW = file["conditions"]
    events = file["ntuple"]
    if isMC==1:
        sumW = 0
        SumWeights = GENW["genEventSumw"].array(library="np")
        sumW = np.sum(SumWeights)
        #if isSignal==1:
        #    sumW = events.num_entries
    else:
        sumW = 1

    if isMC:
        if isSignal:
            weight = float(xs_sig[ShortName])/sumW*lumi
        else:
            weight = float(xs_bkg[ShortName])/sumW*lumi
    else:
        weight = 1.0

    print("weight: ", weight)

    data = {}
    data = events.arrays(variables, library="np")
    #data["weight"] = weight
    data["weight"] = np.full(len(data["event"]), weight)
    print("%i events... "%(len(data["event"])),end='',flush=True)
    #initial_events += data.shape[0]
    #data = preprocess(data)

    #data = preselect(data) #TODO add cutflow
    #data = decorate_hmm(data)
    data = select(data)
    if isMC:
        data["eventWeight"] = data["genWeight"]*data["puweight"]*data["weight"]*data["id_wgt_mu_1"]*data["id_wgt_mu_2"]*data["iso_wgt_mu_1"]*data["iso_wgt_mu_2"]
    else:
        #data["eventWeight"] = np.full(len(data["event"]), data["weight"])
        data["eventWeight"] = data["weight"]
    #initial_events += data.shape[0]
    #final_events += data.shape[0]
    #final_events += len(next(iter(data.values())))
    data['nleptons'] = data['nelectrons'] + data['nmuons']
    #data_zero_jet_ = data[(data['njets'] == 0) & (data['nleptons']  < 3)]
    #data_one_jet_ = data[(data['njets']  == 1) & (data['nleptons']  < 3)]
    #data_two_jet_ = data[(data['njets']  >= 2) & (data['nleptons']  < 3)]
    #data_zero_to_one_jet_ = data[(data['njets']  < 2) & (data['nleptons']  < 3)]
    #data_VH_ttH_ =  data[data['nleptons']  >= 3]
    #data_zero_jet = ak.to_pandas(data_zero_jet_)
    #data_one_jet = ak.to_pandas(data_one_jet_)
    #data_two_jet = ak.to_pandas(data_two_jet_)
    #data_zero_to_one_jet = ak.to_pandas(data_zero_to_one_jet_)
    #data_VH_ttH = ak.to_pandas(data_VH_ttH_)
    #data_zero_jet = data[(data.njets == 0) & (data.nleptons < 3)]
    #data_one_jet = data[(data.njets == 1) & (data.nleptons < 3)]
    #data_two_jet = data[(data.njets >= 2) & (data.nleptons < 3)]
    #data_zero_to_one_jet = data[(data.njets < 2) & (data.nleptons < 3)]
    #data_VH_ttH =  data[data.nleptons >= 3]
    #output['event'] = data['event']
    for branch in branches:
        output[branch] = data[branch]
        #print(branch)
    df = pd.DataFrame(output)
    data_zero_jet = df[(df['njets'] == 0) & (df['nleptons'] < 3)]
    data_one_jet = df[(df['njets'] == 1) & (df['nleptons'] < 3)]
    data_two_jet = df[(df['njets'] >= 2) & (df['nleptons'] < 3)]
    #data_two_jet_m110To150 = df[(df['njets'] >= 2) & (df['nleptons'] < 3) &(df['diMufsr_rc_mass'] > 110) &(df['diMufsr_rc_mass'] < 150) &(df['trg_single_mu24']) ]
    data_two_jet_m110To150 = df[(df['njets'] >= 2) & (df['nleptons'] < 3) &(df['diMufsr_rc_mass'] > 110) &(df['diMufsr_rc_mass'] < 150) &(df['trg_single_mu24']) &(df['nbjets_loose'] <= 1) &(df['nbjets_medium'] <= 0) &(df['nmuons'] == 2) &(df['nelectrons'] == 0) &(df['jet1_eta'] <= 2.4) &(df['jet2_eta'] <= 2.4) &(df['jet1_pt'] >= 35) &(df['jet2_pt'] >= 25) &(df['dijet_mass'] >= 400) &(df['delta_eta_jj'] >= 2.5)]
    data_zero_to_one_jet = df[(df['njets'] < 2) & (df['nleptons'] < 3)]
    data_VH_ttH =  df[df['nleptons'] >= 3]
    with uproot.recreate(args.output) as f:
        #f['inclusive'] = data
        #f['inclusive'] = {var: data[var] for var in branches}
        #f.mktree("passedEvents", branches)
        #f["passedEvents"].extend(output)
        #f['zero_jet'] = {var: data[var] for var in branches}
        f['inclusive'] = output
        f['zero_jet'] = data_zero_jet
        f['one_jet'] = data_one_jet
        f['zero_to_one_jet'] = data_zero_to_one_jet
        f['two_jet'] = data_two_jet
        f['two_jet_m110To150'] = data_two_jet_m110To150
        f['VH_ttH'] = data_VH_ttH
    # data.to_root(args.output, key='inclusive', mode='a', index=False)
    # data_zero_jet.to_root(args.output, key='zero_jet', mode='a', index=False)
    # data_one_jet.to_root(args.output, key='one_jet', mode='a', index=False)
    # data_zero_to_one_jet.to_root(args.output, key='zero_to_one_jet', mode='a', index=False)
    # data_two_jet.to_root(args.output, key='two_jet', mode='a', index=False)
    # data_VH_ttH.to_root(args.output, key='VH_ttH', mode='a', index=False)

    # meta_data = pd.DataFrame({'initial_events': [initial_events], 'final_events': [final_events]})
    # meta_data.to_root(args.output, key='MetaData', mode='a', index=False)

    print("Finished!!! Have gotten the skimmed data in {:s}".format(args.output))

if __name__ == '__main__':
    main()
