input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd/"
input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd_Inc_nbJetsnMuons/"
input_dir = "/publicfs/cms/data/hzz/guoqy/Hmumu/2022/After_hadd_Inc/"
xs_bkg = {}
sumW_bkg = {}

background_samples = []
background_files = {}

def read_sumW(sumW_file):
	file_sumW = open(sumW_file)
	sumW = float(file_sumW.read())
	file_sumW.close
	return sumW

sam = "DYJetsToLL_M50_madgraphMLM"
sam = "DYJetsToLL_M50"
sam_Full = "DYJetsToLL_M-50-madgraphMLM_2022_forPOG"
sam_Full = "DYJetsToLL_M-50-madgraphMLM_2022"
#xs_bkg[sam] = 6225.4
xs_bkg[sam] = 6345.99
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "DYJetsToLL_M50_amcatnloFXFX"
sam = "DYJetsToLL_M50"
sam_Full = "DYto2L-2Jets_MLL-50_amcatnloFXFX_2022_ext1-v1"
xs_bkg[sam] = 6345.99
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "DYto2Mu_MLL-50to120_powheg"
sam_Full = "DYto2Mu_MLL-50to120_powheg_2022"
xs_bkg[sam] = 2219
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "DYto2Mu_MLL-120to200_powheg"
sam_Full = "DYto2Mu_MLL-120to200_powheg_2022"
xs_bkg[sam] = 21.65
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

#sam = "DYJetsToLL_M-105To160"
#sam_Full = "DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX_RunIIAutumn18"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 47.12
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))


#sam = "EWK_LLJJ_M50"
#background_files[sam] = "%sEWK_LLJJ_MLL-50_MJJ-120-madgraph.root"%(input_dir)
#xs_bkg[sam] = 1.029
#sumW_bkg[sam] = read_sumW("%ssumW/EWK_LLJJ_MLL-50_MJJ-120-madgraph.txt"%(input_dir))
#
#sam = "EWK_LLJJ_M105-160"
#background_files[sam] = "%sEWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18.root"%(input_dir)
#xs_bkg[sam] = 0.078
#sumW_bkg[sam] = read_sumW("%ssumW/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18.txt"%(input_dir))
#
#sam = "GGToZZTo2e2mu"
#sam_Full = "GluGluToContinToZZTo2e2mu-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.00329
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "GGToZZTo2e2tau"
#sam_Full = "GluGluToContinToZZTo2e2tau-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.00329
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "GGToZZTo2mu2nu"
#sam_Full = "GluGluToContinToZZTo2mu2nu-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.001772
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "GGToZZTo2mu2tau"
#sam_Full = "GluGluToContinToZZTo2mu2tau-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.00329
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "GGToZZTo4mu"
#sam_Full = "GluGluToContinToZZTo4mu-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.001402
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "GGToZZTo4tau"
#sam_Full = "GluGluToContinToZZTo4tau-mcfm701"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 0.001402
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
sam = "ZZTo4L"
sam_Full = "ZZto4L_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 1.65
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "ZZTo2L2Nu"
sam_Full = "ZZto2L2Nu_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 1.19
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WWTo2L2Nu"
sam_Full = "WWto2L2Nu_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 12.98
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WZTo3LNu"
sam_Full = "WZto3LNu_powheg_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] =  5.31
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "ZZTo2L2Q"
sam_Full = "ZZto2L2Q_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 8.08
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WZTo2L2Q"
sam_Full = "WZto2L2Q_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 8.17
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WWW"
sam_Full = "WWW_4F_amcatnlo-madspin_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.2328
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WWZ"
sam_Full = "WWZ_4F_amcatnlo_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.1851
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "WZZ"
sam_Full = "WZZ_amcatnlo_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.06206
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "ZZZ"
sam_Full = "ZZZ_amcatnlo_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.01591
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "TTTo2L2Nu"
sam_Full = "TTto2L2Nu_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 97.4488
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

#sam = "TTToSemiLep"
#sam_Full = "TTToSemiLeptonic-powheg"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 358.57
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "ST_tW_top"
#sam_Full = "ST_tW_top_5f_inclusiveDecays-powheg"
sam_Full = "TWminusto2L2Nu_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 35.99
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "ST_tW_antitop"
#sam_Full = "ST_tW_antitop_5f_inclusiveDecays-powheg"
sam_Full = "TbarWplusto2L2Nu_powheg_2022_ext1-v2"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 36.05
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

#sam = "ST_t-channel_top"
#sam_Full = "ST_t-channel_top_5f_InclusiveDecays-powheg"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 136.02
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "ST_t-channel_antitop"
#sam_Full = "ST_t-channel_antitop_5f_InclusiveDecays-powheg"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 80.95
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))
#
#sam = "ST_s-channel"
#sam_Full = "ST_s-channel_4f_leptonDecays-amcatnlo"
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 3.40
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

sam = "tZq"
#sam_Full = "tZq_ll_4f_ckm_NLO-madgraph_RunIIAutumn18_ext1-v1"
sam_Full = "TZQB-4FS_OnshellZ_amcatnlo_2022"
background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
xs_bkg[sam] = 0.0801
sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

#good 
#sam = ""
#sam_Full = ""
#background_files[sam] = "%s%s.root"%(input_dir,sam_Full)
#xs_bkg[sam] = 
#sumW_bkg[sam] = read_sumW("%ssumW/%s.txt"%(input_dir,sam_Full))

#good 
#background_samples = ["GGToZZTo2e2mu","GGToZZTo2mu2nu","GGToZZTo2mu2tau","GGToZZTo4mu","ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","TTTo2L2Nu","TTToSemiLep","ST_tW_top","ST_tW_antitop","ST_t-channel_top","ST_t-channel_antitop","ST_s-channel","tZq","EWK_LLJJ_M50","EWK_LLJJ_M105-160","DYJetsToLL_M-105To160","DYJetsToLL_M50"]
#background_samples = ["ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","WWZ","WZZ","ZZZ","TTTo2L2Nu","ST_tW_top","ST_tW_antitop","tZq","DYJetsToLL_M50"]
background_samples = ["ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","WWZ","WZZ","ZZZ","TTTo2L2Nu","ST_tW_top","ST_tW_antitop","DYJetsToLL_M50"]
#background_samples = ["ZZTo4L","ZZTo2L2Nu","WZTo3LNu","ZZTo2L2Q","WZTo2L2Q","WWTo2L2Nu","WWW","WWZ","WZZ","ZZZ","TTTo2L2Nu","ST_tW_top","ST_tW_antitop","DYto2Mu_MLL-120to200_powheg","DYto2Mu_MLL-50to120_powheg"]

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
"jet1_rawMass",
"jet2_rawMass",
"jet1_rawpT",
"jet2_rawpT",
]
