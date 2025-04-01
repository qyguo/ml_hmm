#!/usr/bin/env python
#
#
#
#  Created by Jay Chan
#
#  8.22.2018
#
#
#
#
#
import os
from ROOT import TFile, TH1F, TH1, gROOT
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import uproot
# from root_pandas import *
import json
from categorizer import *
from pdb import set_trace

pd.options.mode.chained_assignment = None

def getArgs():
    """Get arguments from command line."""
    parser = ArgumentParser()
    parser.add_argument('-r', '--region', action = 'store', choices = ['all_jet', 'two_jet', 'one_jet', 'zero_jet', 'zero_to_one_jet', 'VH_ttH'], default = 'two_jet', help = 'Region to process')
    parser.add_argument('-i', '--input', action = 'store', default = '/afs/cern.ch/work/q/qguo/Hmumu/ML_hmm/hzgml/outputs_0624/', help = 'Path of root file for categorization')
    parser.add_argument('-n', '--nscan', type = int, default = 100, help='number of scan.')
    parser.add_argument('-b', '--nbin', type = int, default = 10, choices = [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16], help = 'number of BDT bins.')
    parser.add_argument('--skip', action = 'store_true', default = False, help = 'skip the hadd part')
    parser.add_argument('--minN', type = float, default = 10, help = 'minimum number of events in mass window')
    parser.add_argument('-v', '--variable', action = 'store', choices = ['bdt', 'NN'], default = 'bdt', help = 'MVA variable to use')
    #parser.add_argument('--val', action = 'store_true', default = False, help = 'se validation samples for categroization')
    #parser.add_argument('-t', '--transform', type = bool, default = True, help = 'use the transform scores for categroization')
    parser.add_argument('-t', '--transform', action='store_true', default=False, help = 'use the transform scores for categroization')
    parser.add_argument('-ar', '--arctanh_', action='store_true', default=False, help = 'use the arctanh scores for categroization')
    parser.add_argument('--floatB', action = 'store_true', default = False, help = 'Floting last boundary')
    parser.add_argument('-es', '--estimate', action = 'store', choices = ['fullSim', 'fullSimrw', 'data_sid'], default = 'fullSim', help = 'Method to estimate significance')

    parser.add_argument('-f', '--nfold', type = int, default = 1, help='number of folds.')
    parser.add_argument('-e', '--earlystop', type = int, default = -1, help='early stopping rounds.')

    parser.add_argument('-s', '--shield', action='store', type=int, default=-1, help='Which variables needs to be shielded')
    parser.add_argument('-a', '--add', action='store', type=int, default=-1, help='Which variables needs to be added')

    return  parser.parse_args()

def gettingsig(input_path, region, variable, boundaries, transform, arctanh_, estimate):

    nbin = len(boundaries[0])

    yields = pd.DataFrame({'sig': [0.]*nbin,
                          'sig_err': [0.]*nbin,
                          'sig_tot': [0.]*nbin,
                          'sig_tot_err': [0.]*nbin,
                          'data_sid': [0.]*nbin,
                          'data_sid_err': [0.]*nbin,
                          'bkgmc_sid': [0.]*nbin,
                          'bkgmc_sid_err': [0.]*nbin,
                          'bkgmc_cen': [0.]*nbin,
                          'bkgmc_cen_err': [0.]*nbin,
                          'vbf': [0.]*nbin,
                          'vbf_err': [0.]*nbin,
                          'ggH': [0.]*nbin,
                          'ggH_err': [0.]*nbin})

    #for category in ['sig', 'vbf', "data_sid", "bkgmc_sid", "bkgmc_cen", "sig_tot"]:#['sig', 'vbf', 'ggH', "data_sid", "bkgmc_sid", "bkgmc_cen", "sig_tot"]
    for category in ['sig', 'vbf', "bkgmc_sid", "bkgmc_cen", "sig_tot"]:#['sig', 'vbf', 'ggH', "data_sid", "bkgmc_sid", "bkgmc_cen", "sig_tot"]
        # for data in tqdm(read_root(f'{input_path}/{region}/{"bkgmc" if "bkgmc" in category else "data" if "data" in category else "sig" if "sig" in category else category}.root', key='test', columns=[f"{variable}_score{'_t' if transform else ''}", 'H_mass', 'eventWeight', 'event'], chunksize=500000), desc=f'Loading {category}', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
        # Define the file path
        #file_path = f'{input_path}/{region}/{"bkgmc" if "bkgmc" in category else "data" if "data" in category else "sig" if "sig" in category else category}.root'
        file_path = f'{input_path}/{region}/{"bkgmc" if "bkgmc" in category else "data" if "data" in category else "sig" if "sig" in category else "VBFHToMuMu_M125" if "vbf" in category else category}.root'
        # Open the file
        file = uproot.open(file_path)
        # Define the columns to read
        columns = [f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}", 'H_mass', 'eventWeight', 'event']

        # Iterate over the data in chunks
        for data in tqdm(file['test'].iterate(columns, library='pd', step_size=500000), desc=f'Loading {category}', bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):
    
            if 'sid' in category:
                data = data[(data.H_mass >= 110) & (data.H_mass <= 150) & ((data.H_mass < 115) | (data.H_mass > 135))]
                data['w'] = data.eventWeight

            elif 'tot' in category:
                data = data[(data.H_mass >= 110) & (data.H_mass <= 150)]
                data['w'] = data.eventWeight

            else:
                data = data[(data.H_mass >= 115) & (data.H_mass <= 135)]
                data['w'] = data.eventWeight
           
            # print(data)
    
            for i in range(len(boundaries)):
    
                data_s = data[data.event % len(boundaries) == i]
    
                for j in range(len(boundaries[i])):
    
                    data_ss = data_s[data_s[f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}"] >= boundaries[i][j]]
                    if j != len(boundaries[i]) - 1: data_ss = data_ss[data_ss[f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}"] < boundaries[i][j+1]]
    
                    yields[category][j] += data_ss.w.sum()
                    yields[category+'_err'][j] = np.sqrt(yields[category+'_err'][j]**2 + (data_ss.w**2).sum())

    if estimate == "data_sid":
        yields['bkg'] = yields['data_sid']*0.20
        yields['bkg_err'] = yields['data_sid_err']*0.20
    elif estimate == "fullSimrw":
        yields['bkg'] = yields['data_sid']*yields['bkgmc_cen']/yields['bkgmc_sid']
        yields['bkg_err'] = yields['data_sid_err']*yields['bkgmc_cen']/yields['bkgmc_sid']
    elif estimate == "fullSim":
        yields['bkg'] = yields['bkgmc_cen']
        yields['bkg_err'] = yields['bkgmc_cen_err']

    zs = calc_sig(yields.sig, yields.bkg, yields.sig_err, yields.bkg_err)
    yields['z'] = zs[0]
    yields['u'] = zs[1]
    yields['VBF purity [%]'] = yields['vbf']/yields['sig']*100

    for i in yields:
        print(yields[i])

    z = np.sqrt((yields['z']**2).sum())
    u = np.sqrt((yields['z']**2 * yields['u']**2).sum())/z

    print(f'Significance:  {z:.4f} +/- {abs(u):.4f}')

    return z, abs(u), yields

def categorizing(input_path, region, variable, sigs, bkgs, nscan, minN, transform, arctanh_, nbin, floatB, n_fold, fold, earlystop, estimate):

    f_sig = TFile('%s/%s/sig.root' % (input_path, region))
    t_sig = f_sig.Get('test')
 
    if estimate in ["fullSim", "fullSimrw"]:
        f_bkgmc = TFile('%s/%s/bkgmc.root' % (input_path, region))
        t_bkgmc = f_bkgmc.Get('test')
    if estimate in ["fullSimrw", "data_sid"]:
        f_data_sid = TFile('%s/%s/data.root' % (input_path, region))
        t_data_sid = f_data_sid.Get('test')

    h_sig = TH1F('h_sig','h_sig',nscan,0,1)
    h_bkg = TH1F('h_bkg','h_bkg',nscan,0,1)

    t_sig.Draw(f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}>>h_sig", "eventWeight*%f*((H_mass>=115&&H_mass<=135)&&(event%%%d!=%d))"%(n_fold/(n_fold-1.) if n_fold != 1 else 1, n_fold, fold if n_fold != 1 else 1))

    # filling bkg histograms
    if estimate in ["fullSim", "fullSimrw"]:
        h_bkgmc_cen = TH1F('h_bkgmc_cen', 'h_bkgmc_cen', nscan, 0., 1.)
        t_bkgmc.Draw(f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}>>h_bkgmc_cen", "eventWeight*%f*((H_mass>=115&&H_mass<=135)&&(event%%%d!=%d))"%(n_fold/(n_fold-1.) if n_fold != 1 else 1, n_fold, fold if n_fold != 1 else 1))
    if estimate in ["fullSimrw"]:
        h_bkgmc_sid = TH1F('h_bkgmc_sid', 'h_bkgmc_sid', nscan, 0., 1.)
        t_bkgmc.Draw(f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}>>h_bkgmc_sid", "eventWeight*%f*((H_mass>=100&&H_mass<=180)&&!(H_mass>=115&&H_mass<=135)&&(event%%%d!=%d))"%(n_fold/(n_fold-1.) if n_fold != 1 else 1, n_fold, fold if n_fold != 1 else 1))
    if estimate in ["fullSimrw", "data_sid"]:
        h_data_sid = TH1F('h_data_sid', 'h_data_sid', nscan, 0., 1.)
        t_data_sid.Draw(f"{variable}_score{'_t' if transform else ''}{'_arctanh' if arctanh_ else ''}>>h_data_sid", "eventWeight*%f*((H_mass>=100&&H_mass<=180)&&!(H_mass>=115&&H_mass<=135)&&(event%%%d!=%d))"%(n_fold/(n_fold-1.) if n_fold != 1 else 1, n_fold, fold if n_fold != 1 else 1))

    if estimate == "data_sid":
        h_data_sid.Scale(0.20)
        cgz = categorizer(h_sig, h_data_sid)
    elif estimate == "fullSimrw":
        cgz = categorizer(h_sig, h_bkgmc_cen, h_bkg_rw_num=h_data_sid, h_bkg_rw_den=h_bkgmc_sid)
    elif estimate == "fullSim":
        cgz = categorizer(h_sig, h_bkgmc_cen)
    #cgz.smooth(60, nscan)  #uncomment this line to fit a function to the BDT distribution. Usage: categorizer.smooth(left_bin_to_fit, right_bin_to_fit, SorB='S' (for signal) or 'B' (for bkg), function='Epoly2', printMessage=False (switch to "True" to print message))
    bmax, zmax = cgz.fit(1, nscan, nbin, minN=minN, floatB=floatB, earlystop=earlystop, pbar=True)
    print(bmax)
    boundaries = bmax
    boundaries_values = [(i-1.)/nscan for i in boundaries]
    print('=========================================================================')
    print(f'Fold number {fold}')
    print(f'The maximal significance:  {zmax}')
    print('Boundaries: ', boundaries_values)
    print('=========================================================================')

    return boundaries, boundaries_values, zmax
    


def main():

    gROOT.SetBatch(True)
    TH1.SetDefaultSumw2(1)

    args=getArgs()
    shield = args.shield
    add = args.add

    input_path = args.input

    #sigs = ["vbf"]
    #sigs = ["VBFHToMuMu_M125", "GluGluHToMuMu_M125"]
    sigs = ["VBFHToMuMu_M125"]

    # bkgs = ['data_fake', 'mc_med', 'mc_true']
    #bkgs = ["dy50"]
    #bkgs = ["DY_105To160", "DY_105To160_VBFFilter", "DY_50FxFx", "DY_50MLM", "EWK_LLJJ_M105To160", "sig", "ST_s-channel", "ST_t-channel_antitop", "ST_t-channel_top", "ST_tW_antitop", "ST_tW_top", "TTTo2L2Nu", "TTToSemiLep", "tZq", "VBFHToMuMu_M125"]
    #UL18
    bkgs = ["DY_105To160", "EWK_LLJJ_M105To160", "ST_s-channel", "ST_t-channel_antitop", "ST_t-channel_top", "ST_tW_antitop", "ST_tW_top", "TTTo2L2Nu", "TTToSemiLep", "tZq"]
    #2022EE
    bkgs = ["DY_105To160","ST_tW_top","ST_tW_top","TTTo2L2Nu"]
    #bkgs = ["DY_105To160","ST_tW_top","ST_tW_top"]

    #bkgs = ["DY_50FxFx", "EWK_LLJJ_50", "ST_s-channel", "ST_t-channel_antitop", "ST_t-channel_top", "ST_tW_antitop", "ST_tW_top", "TTTo2L2Nu", "TTToSemiLep", "tZq"]
    #bkgs = ["DY_50MLM", "EWK_LLJJ_50", "ST_s-channel", "ST_t-channel_antitop", "ST_t-channel_top", "ST_tW_antitop", "ST_tW_top", "TTTo2L2Nu", "TTToSemiLep", "tZq"]
    if args.floatB and args.nbin == 16:
        print('ERROR: With floatB option, the maximun nbin is 15!!')
        quit()


    region = args.region

    nscan=args.nscan

    variable = args.variable

    if not args.skip:
        siglist=''
        for sig in sigs:
            if os.path.isfile('%s/%s/%s.root'% (input_path, region,sig)): siglist+=' %s/%s/%s.root'% (input_path, region,sig)
        os.system("hadd -f %s/%s/sig.root"%(input_path, region)+siglist)

    if not args.skip:
        bkglist=''
        for bkg in bkgs:
            if os.path.isfile('%s/%s/%s.root'% (input_path, region,bkg)): bkglist+=' %s/%s/%s.root'% (input_path,region,bkg)
        os.system("hadd -f %s/%s/bkgmc.root"%(input_path, region)+bkglist)

    print("args.transform: ", args.transform)
    print("args.arctanh_: ", args.arctanh_)

    n_fold = args.nfold
    boundaries=[]
    boundaries_values=[]
    smaxs = []
    for j in range(n_fold):
        bound, bound_value, smax = categorizing(input_path, region, variable, sigs, bkgs, nscan, args.minN, args.transform, args.arctanh_, args.nbin if not args.floatB else args.nbin + 1, args.floatB, n_fold, j, args.earlystop, estimate=args.estimate)
        boundaries.append(bound)
        boundaries_values.append(bound_value)
        smaxs.append(smax)

    # boundaries_values.append([0.00, 0.29, 0.57, 0.73])

    smax = sum(smaxs)/n_fold
    print('Averaged significance: ', smax)

    s, u, yields = gettingsig(input_path, region, variable, boundaries_values, args.transform, args.arctanh_, estimate=args.estimate)

    outs={}
    outs['boundaries'] = boundaries
    outs['boundaries_values'] = boundaries_values
    outs['smax'] = smax
    outs['significance'] = s
    outs['Delta_significance'] = u
    outs['nscan'] = nscan
    outs['transform'] = args.transform
    outs['arctanh_'] = args.arctanh_
    outs['floatB'] = args.floatB
    outs['nfold'] = n_fold
    outs['minN'] = args.minN
    outs['fine_tuned'] = False
    outs['variable'] = variable
    outs['estimate'] = args.estimate

    print(outs, '\n================================================\n')

    if not os.path.isdir('%s/significances/%s'%(input_path,region)):
        print(f'INFO: Creating output folder: "{input_path}/significances/{region}"')
        os.makedirs("%s/significances/%s"%(input_path,region))

    with open('%s/significances/bin_binaries_1D_%s.txt'%(input_path,region), 'w') as json_file:
        json_file.write('1\n1\n')
        for i in boundaries_values:
            json_file.write('{:d} '.format(len(i)))
            for j in i:
                json_file.write('{:.2f} '.format(j))
        json_file.write('1.00\n')
        for i in list(yields['z']):
            json_file.write('{:.4f} '.format(i))
        json_file.write('%.4f %.4f' % (s, u))
    with open('%s/significances/%d_%d_%s_1D_%d.json' % (input_path, shield+1, add+1, region, args.nbin), 'w') as json_file:
        json.dump(outs, json_file)
        for i in list(yields['z']):
            json_file.write('{:.4f}\n'.format(i))
        json_file.write('{:.4f}'.format(s))

    return



if __name__ == '__main__':
    main()
