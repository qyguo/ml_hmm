input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/UL18/After_hadd_check/"

data_samples = ["data"]
data_files = {"data": "%sSingleMuon_UL18.root"%(input_dir)}

data_vars = [
"Flag_DiMuonFromHiggs",
"Flag_LeptonChargeSumVeto",
"Flag_dimuon_Zmass_veto",
"H_eta",
"H_mass",
"H_phi",
"H_pt",
"dijet_eta",
"dijet_mass",
"event",
"is_data",
"is_diboson",
"is_dyjets",
"is_embedding",
"is_gghmm",
"is_top",
"is_triboson",
"is_vbfhmm",
"is_vhmm",
"is_wjets",
"is_zjjew",
"jet1_eta",
"jet1_mass",
"jet1_phi",
"jet1_pt",
"jet2_eta",
"jet2_mass",
"jet2_phi",
"jet2_pt",
"lumi",
"met_phi",
"met_pt",
"mu1_fromH_eta",
"mu1_fromH_phi",
"mu1_fromH_pt",
"mu1_mu2_dphi",
"mu2_fromH_eta",
"mu2_fromH_phi",
"mu2_fromH_pt",
"mumuH_dR",
"nbjets_loose",
"nbjets_medium",
"nelectrons",
"njets",
"nmuons",
"run",
"trg_single_mu24",
#"trg_single_mu27",
#"id_wgt_mu_1",
#"id_wgt_mu_2",
#"iso_wgt_mu_1",
#"iso_wgt_mu_2",
"mu1_fromH_ptErr",
"mu2_fromH_ptErr",
"pt_rc_1",
"pt_rc_2",
#"prefiring_wgt",
"fsrPhoton_pt_1",
"fsrPhoton_eta_1",
"fsrPhoton_phi_1",
"fsrPhoton_dROverEt2_1",
"fsrPhoton_relIso03_1",
"fsrPhoton_pt_2",
"fsrPhoton_eta_2",
"fsrPhoton_phi_2",
"fsrPhoton_dROverEt2_2",
"fsrPhoton_relIso03_2",
"SoftActivityJetNjets5",
"jet1_qgl",
"jet2_qgl",
]
