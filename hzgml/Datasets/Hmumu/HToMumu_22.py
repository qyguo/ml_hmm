input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd/"
input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd_Inc_nbJetsnMuons/"
input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd_Inc/"

masses = [125]
#prod = ["GluGluHToMuMu","VBFHToMuMu","WplusH HToMuMu","WminusH HToMuMu","ZH HToMuMu","ttHToMuMu"]
prod = ["GluGluHto2Mu","VBFHto2Mu"]
xs_sig_new = { "GluGluHto2Mu_M-125": 0.0114359,
               "VBFHto2Mu_M-125": 0.000884275,
}
xs_err_new = { "GluGluHto2Mu_M-125": 0.0,
               "VBFHto2Mu_M-125": 0.0,
}
#xs_sig_new = {	"GluGluHToMuMu_M125": 0.0114359,
#		"GluGluHToMuMu_M120": 0.,
#		"GluGluHToMuMu_M130": 0.,
#		"VBFHToMuMu_M125": 0.000884275,
#		"VBFHToMuMu_M120": 0.000,
#		"VBFHToMuMu_M130": 0.000,
#		"WplusH HToMuMu WToAll M125": 0.000,
#		"WplusH HToMuMu WToAll M120": 0.000,
#		"WplusH HToMuMu WToAll M130": 0.000,
#		"WminusH HToMuMu WToAll M125": 0.000,
#		"WminusH HToMuMu WToAll M120": 0.000,
#		"WminusH HToMuMu WToAll M130": 0.0000,
#		"ZH HToMuMu ZToAll M125": 0.000,
#		"ZH HToMuMu ZToAll M120": 0.000,
#		"ZH HToMuMu ZToAll M130": 0.000,
#		"ttHToMuMu M125": 0.000,
#		"ttHToMuMu M120": 0.000,
#		"ttHToMuMu M130": 0.0000,
#}
#xs_err_new = {	"GluGluHToMuMu_M125": 0.0,
#		"GluGluHToMuMu_M120": 0.0,
#		"GluGluHToMuMu_M130": 0.0,
#		"VBFHToMuMu_M125": 0.0,
#		"VBFHToMuMu_M120": 0.0,
#		"VBFHToMuMu_M130": 0.0,
#		"WplusH HToMuMu WToAll M125": 0.0,
#		"WplusH HToMuMu WToAll M120": 0.0,
#		"WplusH HToMuMu WToAll M130": 0.0,
#		"WminusH HToMuMu WToAll M125": 0.0,
#		"WminusH HToMuMu WToAll M120": 0.0,
#		"WminusH HToMuMu WToAll M130": 0.0,
#		"ZH HToMuMu ZToAll M125": 0.0,
#		"ZH HToMuMu ZToAll M120": 0.0,
#		"ZH HToMuMu ZToAll M130": 0.0,
#		"ttHToMuMu M125": 0.0,
#		"ttHToMuMu M120": 0.0,
#		"ttHToMuMu M130": 0.0,
#}
xs_sig = {}
xs_err = {}
sumW_sig = {}
for p in prod:
    for m in masses:
        xs_sig["%s_M-%i"%(p,m)] = xs_sig_new["%s_M-%i"%(p,m)]
        xs_err["%s_M-%i"%(p,m)] = xs_err_new["%s_M-%i"%(p,m)]

def read_sumW(sumW_file):
    file_sumW = open(sumW_file)
    sumW = float(file_sumW.read())
    file_sumW.close
    return sumW

signal_samples = []
signal_files = {}
for p in prod:
    for m in masses:
        signal_samples.append("%s_M-%i"%(p,m))
        signal_files["%s_M-%i"%(p,m)] = "%s%s_M-%i_powheg_2022.root"%(input_dir,p,m)
        xs_sig["%s_M-%i"%(p,m)] *= 20
        xs_err["%s_M-%i"%(p,m)] *= 20
        sumW_sig["%s_M-%i"%(p,m)] = read_sumW("%ssumW/%s_M-%i_powheg_2022.txt"%(input_dir,p,m))
##

signal_vars = [
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
"genWeight",
"genmet_phi",
"genmet_pt",
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
"puweight",
"run",
"trg_single_mu24",
#"trg_single_mu27",
"id_wgt_mu_1",
"id_wgt_mu_2",
"iso_wgt_mu_1",
"iso_wgt_mu_2",
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
"jet1_rawMass",
"jet2_rawMass",
"jet1_rawpT",
"jet2_rawpT",
]
