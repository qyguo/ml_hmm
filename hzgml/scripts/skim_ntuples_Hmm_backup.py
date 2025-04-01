#!/usr/bin/env python
#  Created by Jay Chan
#
#  8.21.2019
#
# Changed 05.22.2024
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
from scripts.ZSelector import *
import awkward as ak
from out_dict import *
#from Datasets.Hmumu.Data_22 import *
#from Datasets.Hmumu.HToMumu_22 import *
#from Datasets.Hmumu.Background_22 import *
from Datasets.Hmumu.Data_UL18 import *
from Datasets.Hmumu.HToMumu_UL18 import *
from Datasets.Hmumu.Background_UL18 import *

def getArgs():
    parser = ArgumentParser(description="Skim the input ntuples for Hmumu XGBoost analysis.")
    parser.add_argument('-i', '--input', action='store', default='inputs', help='Path to the input ntuple')
    parser.add_argument('-o', '--output', action='store', default='outputs', help='Path to the output ntuple')
    parser.add_argument('--chunksize', type=int, default=500000, help='size to process at a time') 
    return  parser.parse_args()

def true_delta_phi(delta_phi):
    if delta_phi > math.pi:
        return 2 * math.pi - delta_phi
    return delta_phi

def compute_is_center(x):

    if (x.H_mass >= 122 and x.H_mass <= 128): return 1
    else: return 0

# others

def compute_Z_cosTheta(x):
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    M, mll = x.H_mass, x.Z_mass
    lZ = math.sqrt((H.Dot(Z) / M) ** 2 - mll ** 2)

    H_transverse_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    H_transverse_beta.SetZ(0)
    hH = Math.VectorUtil.boost(H, -H_transverse_beta)
    hPz, hE = hH.Pz(), hH.E()
    q = Math.LorentzVector("ROOT::Math::PxPyPzE4D<float>")(0, 0, (hPz + hE) / 2, (hE + hPz) / 2)
    q = Math.VectorUtil.boost(q, H_transverse_beta)
    qbar = Math.LorentzVector("ROOT::Math::PxPyPzE4D<float>")(0, 0, (hPz - hE) / 2, (hE - hPz) / 2)
    qbar = Math.VectorUtil.boost(qbar, H_transverse_beta)

    cosTheta = (qbar - q).Dot(Z)/(M * lZ)

    return cosTheta

def compute_l_costheta(x):
    if (x.Z_lead_lepton_charge < 0 and x.Z_sublead_lepton_charge > 0): 
        l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
        l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    elif (x.Z_sublead_lepton_charge < 0 and x.Z_lead_lepton_charge > 0):
        l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
        l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    else: print('leptons have same sign')
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    Z_beta = TLorentzVector(Z.Px(), Z.Py(), Z.Pz(), Z.E()).BoostVector()
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    Z_BH = Math.VectorUtil.boost(Z, -H_beta)
    l1_BZ = Math.VectorUtil.boost(l1, -Z_beta)
    l2_BZ = Math.VectorUtil.boost(l2, -Z_beta)
    gamma_BZ = Math.VectorUtil.boost(gamma, -Z_beta)
    
    ## Z and lepton
    #a = l1_BZ + l2_BZ
    #cosTheta = (a.Vect().Unit()).Dot(l1_BZ.Vect().Unit())
    
    ## formula
    #a = l1_BZ.E() - l2_BZ.E()
    #k1 = math.sqrt(Z_BH.Vect().Dot(Z_BH.Vect()))
    #cosTheta = a/k1

    ## photon and lepton
    cosTheta = - (gamma_BZ.Vect().Unit()).Dot(l1_BZ.Vect().Unit())

    return cosTheta

def compute_l_phi(x):
    if (x.Z_lead_lepton_charge < 0 and x.Z_sublead_lepton_charge > 0): 
        l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
        l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    elif (x.Z_sublead_lepton_charge < 0 and x.Z_lead_lepton_charge > 0):
        l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
        l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    else: print('leptons have same sign')
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    l1_BH = Math.VectorUtil.boost(l1, -H_beta)
    l2_BH = Math.VectorUtil.boost(l2, -H_beta)
    Z_BH = Math.VectorUtil.boost(Z, -H_beta)
    N1_BH = (l1_BH.Vect().Cross(l2_BH.Vect())).Unit()
    beamAxis = TVector3 (0, 0, 1)
    Z3_BH = Z_BH.Vect().Unit()
    NSC_BH = - (Z3_BH.Cross(beamAxis)).Unit()
    # Z_beta = TLorentzVector(Z_BH.Px(), Z_BH.Py(), Z_BH.Pz(), Z_BH.E()).BoostVector()

    tmpSgnPhi1 = Z3_BH.Dot(N1_BH.Cross(NSC_BH))
    sgnPhi1 = 0.
    if (abs(tmpSgnPhi1)>0.): sgnPhi1 = tmpSgnPhi1/abs(tmpSgnPhi1)
    dot_BH1SC = N1_BH.Dot(NSC_BH)
    if (abs(dot_BH1SC)>=1.): dot_BH1SC *= 1./abs(dot_BH1SC)
    Phi1 = sgnPhi1 * math.acos(dot_BH1SC)

    return Phi1

def compute_dR1lg(x):

    if (((x.Z_lead_lepton_eta - x.gamma_eta)**2 + x.Z_lead_lepton_deltaphi**2)**0.5) > ((x.Z_sublead_lepton_eta - x.gamma_eta)**2 + x.Z_sublead_lepton_deltaphi**2)**0.5:
        return ((x.Z_lead_lepton_eta - x.gamma_eta)**2 + x.Z_lead_lepton_deltaphi**2)**0.5
    else:
        return ((x.Z_sublead_lepton_eta - x.gamma_eta)**2 + x.Z_sublead_lepton_deltaphi**2)**0.5
    
def compute_dR2lg(x):

    if (((x.Z_lead_lepton_eta - x.gamma_eta)**2 + x.Z_lead_lepton_deltaphi**2)**0.5) < ((x.Z_sublead_lepton_eta - x.gamma_eta)**2 + x.Z_sublead_lepton_deltaphi**2)**0.5:
        return ((x.Z_lead_lepton_eta - x.gamma_eta)**2 + x.Z_lead_lepton_deltaphi**2)**0.5
    else:
        return ((x.Z_sublead_lepton_eta - x.gamma_eta)**2 + x.Z_sublead_lepton_deltaphi**2)**0.5

# mine

def compute_l_prodAngle(x):

    l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
    l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    z_beta = TLorentzVector(Z.Px(), Z.Py(), Z.Pz(), Z.E()).BoostVector()
    if (x.Z_lead_lepton_charge > 0):
        a = Math.VectorUtil.boost(l1, -z_beta).Vect()
    else :
        a = Math.VectorUtil.boost(l2, -z_beta).Vect()
    return a.Unit().Dot(z_beta.Unit())

def compute_Z_prodAngle(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    Z_BH = Math.VectorUtil.boost(Z, -H_beta).Vect()
    gamma_BH = Math.VectorUtil.boost(gamma, -H_beta).Vect()
    return Z_BH.Unit().Dot(H.Vect().Unit())

def compute_gamma_relEerror(x):

    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    return x.gamma_energyErr / gamma.E()

def compute_G_ECM(x):

    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    return Math.VectorUtil.boost(gamma, -H_beta).E()

def compute_Z_ECM(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    return Math.VectorUtil.boost(Z, -H_beta).E()

def compute_l_rapCM(x):

    l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    z_beta = TLorentzVector(Z.Px(), Z.Py(), Z.Pz(), Z.E()).BoostVector()
    a = Math.VectorUtil.boost(l1, -z_beta)
    return a.Rapidity()

def compute_Z_rapCM(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    H_beta = TLorentzVector(H.Px(), H.Py(), H.Pz(), H.E()).BoostVector()
    a = Math.VectorUtil.boost(Z, -H_beta)
    return a.Rapidity()

def compute_HZ_deltaRap(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    H = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.H_pt, x.H_eta, x.H_phi, x.H_mass)
    return H.Rapidity() - Z.Rapidity()

def compute_ll_deltaR(x):

    l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
    l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    return Math.VectorUtil.DeltaR(l1, l2)

def compute_leadLG_deltaR(x):

    l1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_lead_lepton_pt, x.Z_lead_lepton_eta, x.Z_lead_lepton_phi, x.Z_lead_lepton_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    return Math.VectorUtil.DeltaR(l1, gamma)

def compute_subleadLG_deltaR(x):

    l2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_sublead_lepton_pt, x.Z_sublead_lepton_eta, x.Z_sublead_lepton_phi, x.Z_sublead_lepton_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    return Math.VectorUtil.DeltaR(l2, gamma)

def compute_ZG_deltaR(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)

    return Math.VectorUtil.DeltaR(Z, gamma)

def compute_H_ptt(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)

    return abs(Z.Px() * gamma.Py() - gamma.Px() * Z.Py()) / (Z - gamma).Pt() * 2.0


def compute_H_al(x):
    
    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)

    return (Z.Pt() ** 2 - gamma.Pt() ** 2) / (Z - gamma).Pt()

def compute_H_bt(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    r = abs(Z.Pt()) / abs(gamma.Pt())
    dev = ((Z.Px() - r * gamma.Px()) ** 2 + (Z.Py() - r * gamma.Py()) ** 2) ** 0.5

    return r * abs(Z.Px() * gamma.Py() - gamma.Px() * Z.Py()) / dev * 2.0

def compute_mass_jj(x):

    if x.n_jets < 2:
        return -9999
    else:
        Jets_1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet_1_pt, x.jet_1_eta, x.jet_1_phi, x.jet_1_mass)
        Jets_2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet_2_pt, x.jet_2_eta, x.jet_2_phi, x.jet_2_mass)
        j1j2 = Jets_1+Jets_2
        return j1j2.M()

def compute_delta_eta_jj(x):

    if x.n_jets < 2:
        return -9999
    else:
        return abs(x.jet_1_eta-x.jet_2_eta)

def compute_delta_phi_jj(x):

    if x.n_jets >= 2:
        return true_delta_phi(abs(x.jet_1_phi-x.jet_2_phi))
    return -9999

def compute_delta_phi_zg_jj(x):

    if x.n_jets >= 2:
        Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
        gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
        zg = Z+gamma
        Jets_1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet_1_pt, x.jet_1_eta, x.jet_1_phi, x.jet_1_mass)
        Jets_2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet_2_pt, x.jet_2_eta, x.jet_2_phi, x.jet_2_mass)
        j1j2 = Jets_1+Jets_2
        return true_delta_phi(abs(zg.Phi()-j1j2.Phi()))
    return -9999

def compute_photon_zeppenfeld(x):

    if x.n_jets < 2:
        return -9999
    else:
        return abs(x.gamma_eta-(x.jet_1_eta+x.jet_2_eta)/2)

def compute_H_zeppenfeld(x):

    if x.njets < 2:
        return -9999
    else:
        Mu1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.mu1_fromH_pt, x.mu1_fromH_eta, x.mu1_fromH_phi, x.muon_mass)
        Mu2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.mu2_fromH_pt, x.mu2_fromH_eta, x.mu2_fromH_phi, x.muon_mass)
        Jets_1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet1_pt, x.jet1_eta, x.jet1_phi, x.jet1_mass)
        Jets_2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet2_pt, x.jet2_eta, x.jet2_phi, x.jet2_mass)
        y1 = Jet_1.rapidity
        y2 = Jet_2.rapidity
        y_H = (Mu1 + Mu2).rapidity  # Rapidity of the dimuon system
        return y_H - (y1 + y2) / 2

def compute_z_star_H_zeppenfeld(x):

    if x.njets < 2:
        return -9999
    else:
        H_zeppenfeld = x.H_eta-(x.jet1_eta+x.jet2_eta)/2
        return H_zeppenfeld/abs(x.jet1_eta-x.jet2_eta)

def compute_pt_balance(x):

    if x.njets < 2:
        return -9999
    else:
        Mu1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.mu1_fromH_pt, x.mu1_fromH_eta, x.mu1_fromH_phi, x.muon_mass)
        Mu2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.mu2_fromH_pt, x.mu2_fromH_eta, x.mu2_fromH_phi, x.muon_mass)
        Jets_1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet1_pt, x.jet1_eta, x.jet1_phi, x.jet1_mass)
        Jets_2 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet2_pt, x.jet2_eta, x.jet2_phi, x.jet2_mass)
        total = Mu1+Mu2+Jets_1+Jets_2
        return total.Pt()/(x.mu1_fromH_pt+x.mu2_fromH_pt+x.jet1_pt+x.jet2_pt)

def compute_pt_balance_0j(x):

    Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
    gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
    total = Z+gamma
    return total.Pt()/(x.Z_pt+x.gamma_pt)

def compute_pt_balance_1j(x):

    if x.n_jets < 1:
        return -9999
    else:
        Z = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.Z_pt, x.Z_eta, x.Z_phi, x.Z_mass)
        gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)
        Jets_1 = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.jet_1_pt, x.jet_1_eta, x.jet_1_phi, x.jet_1_mass)
        total = Z+gamma+Jets_1
        return total.Pt()/(x.Z_pt+x.gamma_pt+x.jet_1_pt)

def compute_Delta_Phi(x, var = "gamma_phi", min_jet=0):

    if min_jet:
        if x.n_jets < min_jet: return -9999
        return true_delta_phi(abs(getattr(x, "jet_%d_phi" % min_jet) - x.gamma_phi))
    else:
        return true_delta_phi(abs(getattr(x, var) - x.gamma_phi))

def compute_Delta_R(x, min_jet=0):

    if x.n_jets <= min_jet: return -9999
    else: 
        jet = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(getattr(x, "jet_%d_pt" % min_jet), getattr(x, "jet_%d_eta" % min_jet), getattr(x, "jet_%d_phi" % min_jet), getattr(x, "jet_%d_mass" % min_jet))
        gamma = Math.LorentzVector("ROOT::Math::PtEtaPhiM4D<float>")(x.gamma_pt, x.gamma_eta, x.gamma_phi, x.gamma_mass)

        return Math.VectorUtil.DeltaR(jet, gamma)

# default

def compute_QG(x):

    if x.Jets_jetMultip >= 1 and (abs(x.Jets_Eta_Lead) > 2.1 or x.Jets_PT_Lead < 50):
        Jets_QGscore_Lead, Jets_QGflag_Lead = -1, -1
    else:
        Jets_QGscore_Lead = x.Jets_NTracks_Lead
        Jets_QGflag_Lead = np.heaviside(x.Jets_NTracks_Lead - 11, 0) + np.heaviside(-x.Jets_NTracks_Lead - 9999, -9999)

    if x.Jets_jetMultip >= 2 and (abs(x.Jets_Eta_Sub) > 2.1 or x.Jets_PT_Sub < 50):
        Jets_QGscore_Sub, Jets_QGflag_Sub = -1, -1
    else:
        Jets_QGscore_Sub = x.Jets_NTracks_Sub
        Jets_QGflag_Sub = np.heaviside(x.Jets_NTracks_Sub - 11, 0) + np.heaviside(-x.Jets_NTracks_Sub - 9999, -9999)

    return Jets_QGscore_Lead, Jets_QGflag_Lead, Jets_QGscore_Sub, Jets_QGflag_Sub

def preselect(data):

    #data.query('(Muons_Minv_MuMu_Paper >= 110) | (Event_Paper_Category >= 17)', inplace=True)
    #data.query('Event_Paper_Category > 0', inplace=True)

    return data

#def decorate(data):
#
#    if data.shape[0] == 0: return data
#
#    data['HZ_relM'] = data.H_mass / data.Z_mass
#    data['H_relpt'] = data.H_pt / data.H_mass
#    data['Z_relpt'] = data.Z_pt / data.H_mass
#    data['Z_lead_lepton_relpt'] = data.Z_lead_lepton_pt / data.H_mass
#    data['Z_sublead_lepton_relpt'] = data.Z_sublead_lepton_pt / data.H_mass
#    data['gamma_relpt'] = data.gamma_pt / data.H_mass
#    data['jet_1_relpt'] = data.jet_1_pt / data.H_mass
#    data['jet_2_relpt'] = data.jet_2_pt / data.H_mass
#    data['MET_relpt'] = data.MET_pt / data.H_mass
#    data['gamma_ptRelErr'] = data.apply(lambda x:compute_gamma_relEerror(x), axis=1)
#    data['G_ECM'] = data.apply(lambda x:compute_G_ECM(x), axis=1)
#    data['Z_ECM'] = data.apply(lambda x:compute_Z_ECM(x), axis=1)
#    data['Z_rapCM'] = data.apply(lambda x:compute_Z_rapCM(x), axis=1)
#    data['l_rapCM'] = data.apply(lambda x:compute_l_rapCM(x), axis=1)
#    data['HZ_deltaRap'] = data.apply(lambda x:compute_HZ_deltaRap(x), axis=1)
#    data['l_cosProdAngle'] = data.apply(lambda x:compute_l_prodAngle(x), axis=1)
#    data['Z_cosProdAngle'] = data.apply(lambda x:compute_Z_prodAngle(x), axis=1)
#    data['ll_deltaR'] = data.apply(lambda x:compute_ll_deltaR(x), axis=1)
#    data['leadLG_deltaR'] = data.apply(lambda x:compute_leadLG_deltaR(x), axis=1)
#    data['ZG_deltaR'] = data.apply(lambda x:compute_ZG_deltaR(x), axis=1)
#    data['subleadLG_deltaR'] = data.apply(lambda x:compute_subleadLG_deltaR(x), axis=1)
#    data['H_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'H_phi'), axis=1)
#    data['Z_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_phi'), axis=1)
#    data['Z_lead_lepton_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_lead_lepton_phi'), axis=1)
#    data['Z_sublead_lepton_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'Z_sublead_lepton_phi'), axis=1)
#    for i in np.arange(1,5):
#        data['jet_%d_deltaphi' %i] = data.apply(lambda x: compute_Delta_Phi(x, "jet", min_jet=i), axis=1)
#        data['jet%dG_deltaR' %i] = data.apply(lambda x: compute_Delta_R(x, min_jet=i), axis=1)
#    data['additional_lepton_1_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'additional_lepton_1_phi', min_jet=0), axis=1)
#    data['additional_lepton_2_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'additional_lepton_2_phi', min_jet=0), axis=1) 
#    data['MET_deltaphi'] = data.apply(lambda x: compute_Delta_Phi(x, 'MET_phi'), axis=1)
#    data['weight'] = data.weight_central
#    data['mass_jj'] = data.apply(lambda x: compute_mass_jj(x), axis=1)
#    data['H_ptt'] = data.apply(lambda x: compute_H_ptt(x), axis=1)
#    data['H_al'] = data.apply(lambda x: compute_H_al(x), axis=1)
#    data['H_bt'] = data.apply(lambda x: compute_H_bt(x), axis=1)
#    data['Z_cos_theta'] = data.apply(lambda x:compute_Z_cosTheta(x), axis=1)
#    data['lep_cos_theta'] = data.apply(lambda x: compute_l_costheta(x), axis=1)
#    data['lep_phi'] = data.apply(lambda x: compute_l_phi(x), axis=1)
#    data['l1g_deltaR'] = data.apply(lambda x: compute_dR1lg(x), axis=1) 
#    data['l2g_deltaR'] = data.apply(lambda x: compute_dR2lg(x), axis=1)
#    data['delta_eta_jj'] = data.apply(lambda x: compute_delta_eta_jj(x), axis=1)
#    data['delta_phi_jj'] = data.apply(lambda x: compute_delta_phi_jj(x), axis=1)
#    data['delta_phi_zgjj'] = data.apply(lambda x: compute_delta_phi_zg_jj(x), axis=1)
#    data['photon_zeppenfeld'] = data.apply(lambda x: compute_photon_zeppenfeld(x), axis=1)
#    data['H_zeppenfeld'] = data.apply(lambda x: compute_H_zeppenfeld(x), axis=1)
#    data['pt_balance'] = data.apply(lambda x: compute_pt_balance(x), axis=1)
#    data['pt_balance_0j'] = data.apply(lambda x: compute_pt_balance_0j(x), axis=1)
#    data['pt_balance_1j'] = data.apply(lambda x: compute_pt_balance_1j(x), axis=1)
#    data['is_center'] = data.apply(lambda x: compute_is_center(x), axis=1)
#    #data[['Jets_QGscore_Lead', 'Jets_QGflag_Lead', 'Jets_QGscore_Sub', 'Jets_QGflag_Sub']] = data.apply(lambda x: compute_QG(x), axis=1, result_type='expand')
#
#    #data.rename(columns={'Muons_Minv_MuMu_Paper': 'm_mumu', 'Muons_Minv_MuMu_VH': 'm_mumu_VH', 'EventInfo_EventNumber': 'eventNumber', 'Jets_jetMultip': 'n_j'}, inplace=True)
#    #data.drop(['PassesttHSelection', 'PassesVHSelection', 'GlobalWeight', 'SampleOverlapWeight', 'EventWeight_MCCleaning_5'], axis=1, inplace=True)
#    data = data.astype(float)
#    data = data.astype({'is_center': int, 'Z_lead_lepton_charge': int, 'Z_lead_lepton_id': int, 'Z_sublead_lepton_charge': int, 'Z_sublead_lepton_id': int, "n_jets": int, "n_b_jets": int, "n_leptons": int, "n_electrons": int, "n_muons": int, 'event': int})
#
#    return data



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

    if os.path.isfile(args.output): os.remove(args.output)

    initial_events = 0
    final_events = 0
    isSignal = 0
    isMC = 1
    if "HToMuMu" in args.input or "Hto2Mu" in args.input:
        isSignal = 1
    if "Muon" in args.input:
        isMC = 0
    if !isMC:  variables = data_vars
    elif isSignal:  variables = signal_vars
    else: variables = background_vars

#    for data in tqdm(read_root(args.input, key='DiMuonNtuple', columns=variables, chunksize=args.chunksize), desc='Processing %s' % args.input, bar_format='{desc}: {percentage:3.0f}%|{bar:20}{r_bar}'):

    #data = pd.read_parquet(args.input)
    print("Reading in ",end='',flush=True)
    #file = uproot.open(files[samples[i]])
    file = uproot.open(args.input)
    print("Is signal: ", isSignal)
    print("Is MC: ", isMC)

    GENW = file["conditions"]
    events = file["ntuple"]
    if isMC==1:
        sumW = 0
        SumWeights = GENW["genEventSumw"].array(library="np")
        sumW = np.sum(SumWeights)
        if isSignal==1:
            sumW = events.num_entries
    else:
        sumW = 1
    

    data = {}
    data = events.arrays(variables, library="np")
    #initial_events += data.shape[0]
    #data = preprocess(data)

    data = preselect(data) #TODO add cutflow
    #data = decorate_hmm(data)
    data = select(data)
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
    #for branch in branches:
    #    #output[branch] = np.array([])
    #    output[branch] = data[branch]
    #    print(branch)
    with uproot.recreate(args.output) as f:
        #f['inclusive'] = data
        f['inclusive'] = {var: data[var] for var in branches}
        #f.mktree("passedEvents", branches)
        #f["passedEvents"].extend(output)
        #f['zero_jet'] = data_zero_jet
        #f['one_jet'] = data_one_jet
        #f['zero_to_one_jet'] = data_zero_to_one_jet
        #f['two_jet'] = data_two_jet
        #f['VH_ttH'] = data_VH_ttH
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
