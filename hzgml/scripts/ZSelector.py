import numpy as np
#import awkward as ak
#import uproot_methods
import uproot3_methods
####import pandas as pd
####import joblib
#from tensorflow.keras.models import load_model
#from tensorflow.keras.backend import clear_session
#from sklearn.ensemble import RandomForestClassifier

def calculate_y_star(mu1, mu2, jet1, jet2):
    y1 = jet1.rapidity
    y2 = jet2.rapidity
    y_H = (mu1 + mu2).rapidity  # Rapidity of the dimuon system
    return y_H - (y1 + y2) / 2

def calculate_z_star(y_star, jet1, jet2):
    y1 = jet1.rapidity
    y2 = jet2.rapidity
    return y_star / abs(y1 - y2)

def calculate_R_pT(mu1, mu2, jet1, jet2):
    pT_mumujj = (mu1 + mu2 + jet1 + jet2).pt
    pT_mu_mu = (mu1 + mu2).pt
    return pT_mumujj / (jet1.pt + jet2.pt + pT_mu_mu)

def calculate_R_pT_Mag(mu1, mu2, jet1, jet2):
    # Calculate the transverse momentum vectors manually
    #pTj1 = jet1.to_vector().set_z(0)
    #pTj2 = jet2.to_vector().set_z(0)
    #pT_dimuon = (mu1 + mu2).to_vector().set_z(0)
    jet1_vec = uproot3_methods.TVector3Array(jet1.x, jet1.y, np.zeros_like(jet1.x))
    jet2_vec = uproot3_methods.TVector3Array(jet2.x, jet2.y, np.zeros_like(jet2.x))
    dimuon_vec = uproot3_methods.TVector3Array((mu1+mu2).x, (mu1+mu2).y, np.zeros_like((mu1+mu2).x))
    
    pT_combined = jet1_vec + jet2_vec + dimuon_vec
    #magnitude_manual = np.sqrt(pT_combined.x**2 + pT_combined.y**2)

    # Calculate the combined transverse momentum vector
    #pT_combined = pTj1 + pTj2 + pT_di-mu
    magnitude_manual = pT_combined.mag
    pT_mumujj = (mu1 + mu2 + jet1 + jet2).pt
    #pT_mu_mu = (mu1 + mu2).pt
    #return pT_mumujj / (jet1.pt + jet2.pt + pT_mu_mu)
    return pT_mumujj / magnitude_manual

def calculate_relative_mass_uncertainty(delta_pT_mu1, pT_mu1, delta_pT_mu2, pT_mu2):
    relative_uncertainty = (1/2) * ((delta_pT_mu1 / pT_mu1)**2 + (delta_pT_mu2 / pT_mu2)**2)**0.5
    return relative_uncertainty

def min_delta_eta(mu1, mu2, jet1, jet2):

    dimuon = mu1 + mu2
    eta_dimuon = dimuon.eta

    # Initialize the minimum pseudorapidity difference with a large number
    #min_delta_eta = float('inf')
    min_delta_eta = np.full(mu1.size, np.inf)

    # Calculate the pseudorapidity differences
    delta_eta_jet1 = np.abs(eta_dimuon - jet1.eta)
    delta_eta_jet2 = np.abs(eta_dimuon - jet2.eta)

    # Loop over the jets to calculate the pseudorapidity differences
    #for jet in jets:
    #    delta_eta = abs(eta_dimuon - jet.Eta())
    #    if delta_eta < min_delta_eta:
    #        min_delta_eta = delta_eta
    min_delta_eta = np.minimum(delta_eta_jet1, delta_eta_jet2)

    return min_delta_eta

def select(data):

	data['muon_mass']=0.1056584

	Mu1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['mu1_fromH_pt'],data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	Mu2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['mu2_fromH_pt'],data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])
	jet1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet1_pt'],data['jet1_eta'],data['jet1_phi'],data['jet1_mass'])
	jet2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet2_pt'],data['jet2_eta'],data['jet2_phi'],data['jet2_mass'])

	dijet = jet1 + jet2
	data['dijet_pt'] = dijet.pt
	data['dijet_eta'] = dijet.eta
	data['dijet_phi'] = dijet.phi
	data['dijet_mass'] = dijet.mass
	data['dijet_pt'] = data['dijet_pt'] * (data['njets'] > 1)
	data['dijet_eta'] = data['dijet_eta'] * (data['njets'] > 1)
	data['dijet_phi'] = data['dijet_phi'] * (data['njets'] > 1)
	data['dijet_mass'] = data['dijet_mass'] * (data['njets'] > 1)
	data['dijet_eta'] = np.where(data['njets'] < 2, -100, data['dijet_eta'])
	data['dijet_phi'] = np.where(data['njets'] < 2, -100, data['dijet_phi'])

	#jet1_raw  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet1_rawpT'],data['jet1_eta'],data['jet1_phi'],data['jet1_rawMass'])
	#jet2_raw  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['jet2_rawpT'],data['jet2_eta'],data['jet2_phi'],data['jet2_rawMass'])
	#dijet_raw = jet1_raw + jet2_raw
	#data['dijet_raw_pt'] = dijet_raw.pt
	#data['dijet_raw_eta'] = dijet_raw.eta
	#data['dijet_raw_phi'] = dijet_raw.phi
	#data['dijet_raw_mass'] = dijet_raw.mass
	#data['dijet_raw_pt'] = data['dijet_raw_pt'] * (data['njets'] > 1)
	#data['dijet_raw_eta'] = data['dijet_raw_eta'] * (data['njets'] > 1)
	#data['dijet_raw_phi'] = data['dijet_raw_phi'] * (data['njets'] > 1)
	#data['dijet_raw_mass'] = data['dijet_raw_mass'] * (data['njets'] > 1)

	#data['dijet_raw_eta'] = np.where(data['njets'] < 2, -100, data['dijet_raw_eta'])
	#data['dijet_raw_phi'] = np.where(data['njets'] < 2, -100, data['dijet_raw_phi'])


	data["y_star"] = calculate_y_star(Mu1,Mu2,jet1,jet2)
	data["z_star"] = calculate_z_star(data["y_star"],jet1,jet2)
	#data["log_z_star"] = np.log(data["z_star"])
	data["R_pT_Mag"]   = calculate_R_pT_Mag(Mu1,Mu2,jet1,jet2)
	data["R_pT"]   = calculate_R_pT(Mu1,Mu2,jet1,jet2)
	data["diMu-mass_resolution"]   = calculate_relative_mass_uncertainty(data["mu1_fromH_ptErr"], data["mu1_fromH_pt"], data["mu2_fromH_ptErr"], data["mu2_fromH_pt"])

	Mu1_rc  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_rc_1'],data['mu1_fromH_eta'],data['mu1_fromH_phi'],data['muon_mass'])
	Mu2_rc  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['pt_rc_2'],data['mu2_fromH_eta'],data['mu2_fromH_phi'],data['muon_mass'])
	diMu_rc = Mu1_rc + Mu2_rc
	data['diMu_pt_rc'] = diMu_rc.pt
	data['diMu_eta_rc'] = diMu_rc.eta
	data['diMu_phi_rc'] = diMu_rc.phi
	data['diMu_mass_rc'] = diMu_rc.mass
	data['mu1_pt_diff'] = data['pt_rc_1'] - data['mu1_fromH_pt']
	data['mu2_pt_diff'] = data['pt_rc_2'] - data['mu2_fromH_pt']


	data['fsrPhoton1_tight'] = ( (data['fsrPhoton_dROverEt2_1'] < 0.012) & (data['fsrPhoton_pt_1'] > 0) )
	data['fsrPhoton2_tight'] = ( (data['fsrPhoton_dROverEt2_2'] < 0.012) & (data['fsrPhoton_pt_2'] > 0) )
	data['fsrPhoton_pt_1']  = data['fsrPhoton1_tight'] * data['fsrPhoton_pt_1']
	data['fsrPhoton_eta_1'] = data['fsrPhoton1_tight'] * data['fsrPhoton_eta_1']
	data['fsrPhoton_phi_1'] = data['fsrPhoton1_tight'] * data['fsrPhoton_phi_1']
	data['fsrPhoton_pt_2']  = data['fsrPhoton2_tight'] * data['fsrPhoton_pt_2']
	data['fsrPhoton_eta_2'] = data['fsrPhoton2_tight'] * data['fsrPhoton_eta_2']
	data['fsrPhoton_phi_2'] = data['fsrPhoton2_tight'] * data['fsrPhoton_phi_2']


	fsrPhoton1  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_1'],data['fsrPhoton_eta_1'],data['fsrPhoton_phi_1'],0)
	fsrPhoton2  = uproot3_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(data['fsrPhoton_pt_2'],data['fsrPhoton_eta_2'],data['fsrPhoton_phi_2'],0)

	muonfsr1 = fsrPhoton1 + Mu1
	muonfsr1_rc = fsrPhoton1 + Mu1_rc
	muonfsr2 = fsrPhoton2 + Mu2
	muonfsr2_rc = fsrPhoton2 + Mu2_rc

	diMuonfsr = muonfsr1 + muonfsr2
	diMuonfsr_rc = muonfsr1_rc + muonfsr2_rc

	data['diMufsr_pt']   = (diMuonfsr).pt
	data['diMufsr_eta']  = (diMuonfsr).eta
	data['diMufsr_phi']  = (diMuonfsr).phi
	data['diMufsr_mass'] = (diMuonfsr).mass

	data['diMufsr_rc_pt']   = (diMuonfsr_rc).pt
	data['diMufsr_rc_eta']  = (diMuonfsr_rc).eta
	data['diMufsr_rc_phi']  = (diMuonfsr_rc).phi
	data['diMufsr_rc_mass'] = (diMuonfsr_rc).mass
	data["diMu-mass_resolution_abs"] = data["diMu-mass_resolution"] * data['diMufsr_rc_mass']
	data["log_dijet_mass"] = np.log(data["dijet_mass"])
	data["log_diMufsr_rc_pt"] = np.log(data["diMufsr_rc_pt"])
	data["min_delta_eta_dimu_jets"] = min_delta_eta(Mu1, Mu2, jet1, jet2)
	data["delta_eta_jj"] = np.abs(jet1.eta-jet2.eta)
	
#	# ------------- Select Z for fake rate measurements ------------------------
#
#	data["nGoodMuons"] = goodL1.astype(int) + goodL2.astype(int) + goodL3.astype(int) + goodL4.astype(int)
#
#	Selector_vars = ["etaL1", "phiL1",
#                     "etaL2", "phiL2",
#                     "etaL3", "phiL3",
#                     "dR12", "dR13", "dR23", "dRM0", "m3l", "mt", "met", "nJets",
#                     "M0", "m3l_pt"]
#
#	# --------- Predict discriminator from NN -----------------------
#	#clear_session()
#	#ZModel = load_model("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/ZSelector_model_alt.h5")
#	##Selector_vars = ["dxyL1", "dzL1", "etaL1", "ip3dL1", "phiL1", "sip3dL1", 
#    ##        		 "dxyL2", "dzL2", "etaL2", "ip3dL2", "phiL2", "sip3dL2",
#    ##         		 "dxyL3", "dzL3", "etaL3", "ip3dL3", "phiL3", "sip3dL3",
#    ##         		 "dR12", "dR13", "dR23", "dRM0", "m3l", "mt", "met", "nJets",
#	##				 "M0", "m3l_pt"]
#	#df = pd.DataFrame.from_dict(data)[Selector_vars]
#	#df["sType"], df["Weights"] = 1, 1
#	#maxW = pd.read_pickle("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/maxes.pkl")
#	#minW = pd.read_pickle("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/mins.pkl")
#	#df = (df-minW)/(maxW-minW)
#
#	#data["discriminator"] = (ZModel.predict(df[Selector_vars])).ravel()
#	## ---------------------------------------------------------------
#
#	## --------- Predict Class from Random Forest --------------------
#	#rf = joblib.load("/orange/avery/nikmenendez/Wto3l/Optimizer/MVA/forest_model.joblib")
#	##Selector_vars = ["etaL1", "phiL1",
#    ##                 "etaL2", "phiL2",
#    ##                 "etaL3", "phiL3",
#    ##                 "dR12", "dR13", "dR23", "dRM0", "m3l", "mt", "met", "nJets",
#    ##                 "M0", "m3l_pt"]
#	#df  = pd.DataFrame.from_dict(data)[Selector_vars]
#	#data["forestguess"] = rf.predict(df)
#	## ---------------------------------------------------------------

	return data
