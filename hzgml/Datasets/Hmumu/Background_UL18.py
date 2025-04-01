input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/UL18/After_hadd_check/"
xs_bkg = {}

background_samples = []
background_files = {}


sam = "DYJetsToLL_M50"
sam = "DY_50FxFx"
xs_bkg[sam] = 6225.4
background_files[sam] = "%sDYJetsToLL_M-50-madgraphMLM_20UL18_ext1-v1.root"%(input_dir)

sam = "DYJetsToLL_M50"
sam = "DY_50MLM"
xs_bkg[sam] = 6225.4
background_files[sam] = "%sDYJetsToLL_M-50-amcatnloFXFX_20UL18.root"%(input_dir)

sam = "DY_105To160_VBFFilter"
xs_bkg[sam] = 2.029
background_files[sam] = "%sDYJetsToLL_M-105To160_VBFFilter_TuneCP5_PSweights_13TeV-madgraphMLM_RunIIAutumn18.root"%(input_dir)

sam = "DYJetsToLL_M-105To160"
sam = "DY_105To160"
sam_Full = "DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX_RunIIAutumn18"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 47.12


sam = "EWK_LLJJ_M50"
background_files[sam] = "%sEWK_LLJJ_MLL-50_MJJ-120-madgraph_20UL18.root"%(input_dir)
xs_bkg[sam] = 1.029

sam = "EWK_LLJJ_M105To160"
background_files[sam] = "%sEWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18.root"%(input_dir)
xs_bkg[sam] = 0.078

sam = "GGToZZTo2e2mu"
sam_Full = "GluGluToContinToZZTo2e2mu-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.00329

sam = "GGToZZTo2e2tau"
sam_Full = "GluGluToContinToZZTo2e2tau-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.00329

sam = "GGToZZTo2mu2nu"
sam_Full = "GluGluToContinToZZTo2mu2nu-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.001772

sam = "GGToZZTo2mu2tau"
sam_Full = "GluGluToContinToZZTo2mu2tau-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.00329

sam = "GGToZZTo4mu"
sam_Full = "GluGluToContinToZZTo4mu-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.001402

sam = "GGToZZTo4tau"
sam_Full = "GluGluToContinToZZTo4tau-mcfm701"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.001402

sam = "ZZTo4L"
sam_Full = "ZZTo4L_powheg_pythia8"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 1.325

sam = "ZZTo2L2Nu"
sam_Full = "ZZTo2L2Nu_powheg_pythia8"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.601

#
#sam = "WZTo3LNu"
#sam_Full = "WZTo3LNu_mllmin4p0-powheg"
#background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 4.658
#

sam = "WWTo2L2Nu"
sam_Full = "WWTo2L2Nu-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 12.178

sam = "WZTo3LNu"
sam_Full = "WZTo3LNu-powheg_RunIIAutumn18_ext1-v1"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] =  4.658

sam = "ZZTo2L2Q"
sam_Full = "ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 3.688

sam = "WZTo2L2Q"
sam_Full = "WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 6.331

sam = "WWW"
sam_Full = "WWW_4F-amcatnlo"
background_files[sam] = "%s%s_20UL18_ext1-v2.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.2086

sam = "WWZ"
sam_Full = "WWZ_4F-amcatnlo"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.1651

sam = "WZZ"
sam_Full = "WZZ-amcatnlo"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.05565

sam = "ZZZ"
sam_Full = "ZZZ-amcatnlo"
background_files[sam] = "%s%s_20UL18_ext1-v2.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.01398

sam = "TTTo2L2Nu"
sam_Full = "TTTo2L2Nu-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 86.61

sam = "TTToSemiLep"
sam_Full = "TTToSemiLeptonic-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 358.57

sam = "ST_tW_top"
sam_Full = "ST_tW_top_5f_inclusiveDecays-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 35.9

sam = "ST_tW_antitop"
sam_Full = "ST_tW_antitop_5f_inclusiveDecays-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 35.9

sam = "ST_t-channel_top"
sam_Full = "ST_t-channel_top_5f_InclusiveDecays-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 136.02

sam = "ST_t-channel_antitop"
sam_Full = "ST_t-channel_antitop_5f_InclusiveDecays-powheg"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 80.95

sam = "ST_s-channel"
sam_Full = "ST_s-channel_4f_leptonDecays-amcatnlo"
background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
xs_bkg[sam] = 3.40

sam = "tZq"
sam_Full = "tZq_ll_4f_ckm_NLO-madgraph_RunIIAutumn18_ext1-v1"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.0758

#good 
#sam = ""
#sam_Full = ""
#background_files[sam] = "%s%s_20UL18.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 

##ok
##background_samples = ["GGToZZTo2e2mu","GGToZZTo2mu2nu","GGToZZTo2mu2tau","GGToZZTo4mu","ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","TTTo2L2Nu","TTToSemiLep","ST_tW_top","ST_tW_antitop","ST_t-channel_top","ST_t-channel_antitop","ST_s-channel","tZq","EWK_LLJJ","DYJetsToLL_M50"]
##all
#background_samples = ["GGToZZTo2e2mu","GGToZZTo2mu2nu","GGToZZTo2mu2tau","GGToZZTo4mu","ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","TTTo2L2Nu","TTToSemiLep","ST_tW_top","ST_tW_antitop","ST_t-channel_top","ST_t-channel_antitop","ST_s-channel","tZq","EWK_LLJJ_M50","EWK_LLJJ_M105-160","DYJetsToLL_M-105To160","DYJetsToLL_M50"]
# 76-106
background_samples = ["GGToZZTo2e2mu","GGToZZTo2mu2nu","GGToZZTo2mu2tau","GGToZZTo4mu","ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","TTTo2L2Nu","TTToSemiLep","ST_tW_top","ST_tW_antitop","ST_t-channel_top","ST_t-channel_antitop","ST_s-channel","tZq","EWK_LLJJ_M50","DYJetsToLL_M50"]
# 110-160
#background_samples = ["GGToZZTo2e2mu","GGToZZTo2mu2nu","GGToZZTo2mu2tau","GGToZZTo4mu","ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","TTTo2L2Nu","TTToSemiLep","ST_tW_top","ST_tW_antitop","ST_t-channel_top","ST_t-channel_antitop","ST_s-channel","tZq","EWK_LLJJ_M105-160","DYJetsToLL_M-105To160"]

background_vars = [
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
"SoftActivityJetNjets5",
"jet1_qgl",
"jet2_qgl",
]
