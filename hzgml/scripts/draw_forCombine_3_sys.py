import ROOT
import array

def create_histogram(file_paths, tree_name, var_to_cut, cut_range, var_to_plot, hist_name, bin_edges, Weights):
    bin_array = array.array('d', bin_edges)
    hist = ROOT.TH1F(hist_name, f"Distribution of {var_to_plot}", len(bin_edges)-1, bin_array)
    hist.Sumw2()

    for file_path in file_paths:
        print(f"Opening file: {file_path}")
        file = ROOT.TFile.Open(file_path)
        if not file or not file.IsOpen():
            print(f"Failed to open file: {file_path}")
            continue

        tree = file.Get(tree_name)
        if not tree:
            print(f"Tree '{tree_name}' not found in {file_path}")
            file.Close()
            continue

        print(f"File {file_path} opened successfully, processing {tree.GetEntries()} entries.")

        # Print example entries to debug
        #tree.Show(0)  # Print the first entry
        #tree.Scan("bdt_score_t:diMufsr_rc_mass", "", "colsize=10 col=bdt_score_t diMufsr_rc_mass")  # Show values

        # Define cut and draw command
        cut_str = f"{Weights}*({var_to_cut} >= {cut_range[0]} && {var_to_cut} <= {cut_range[1]})"
        temp_hist_name = f"temp_{hist_name}"
        temp_hist = ROOT.TH1F(temp_hist_name, f"Temp Histogram for {var_to_plot}", len(bin_edges)-1, bin_array)
        temp_hist.Sumw2()

        draw_cmd = f"{var_to_plot} >> {temp_hist_name}"
        print(f"Draw command: {draw_cmd}, Cut: {cut_str}")

        tree.Draw(draw_cmd, cut_str, "goff")
        print(f"Integral of temporary histogram: {temp_hist.Integral()}")

        # Add temporary histogram contents to main histogram
        hist.Add(temp_hist)

        file.Close()
        print(f"Integral after adding from {file_path}: {hist.Integral()}")

    return hist

def main():
    # bdt
    bin_edges = [0.0, 0.06, 0.12, 0.2, 0.3, 0.41, 0.49, 0.58, 0.64, 0.78, 1.0] #105To160
    bin_edges = [0.0, 0.05, 0.15, 0.23, 0.31, 0.4, 0.47, 0.53, 0.59, 0.71, 1.0] #105To160 noqgl
    #bin_edges = [0.0, 0.12, 0.23, 0.26, 0.31, 0.37, 0.4, 0.5, 0.51, 0.55, 1.0] #50MLM
    #bin_edges =[0.0, 0.12, 0.3, 0.36, 0.37, 0.42, 0.43, 0.5, 0.55, 0.66, 1.0]
    # dnn
    #bin_edges = [0.0, 0.09, 0.16, 0.2, 0.28, 0.55, 0.64, 0.72, 0.88, 0.96, 1.0]
    #bin_edges = [0.0, 0.52, 0.59, 0.63, 0.71, 0.73, 0.74, 0.77, 0.82, 0.92, 1.0]
    #bin_edges = [0.0, 0.58, 0.67, 0.77, 0.81, 0.83, 0.87, 0.92, 0.95, 1.0]#used
    # BDT with jet1_qgl jet2_qgl
    #path_ = "outputs_0624/two_jet/"
    #path_ = "outputs_0624_SB_HM125/two_jet/"
    # BDT without jet1_qgl jet2_qgl
    #path_ = "outputs_0730/two_jet/"
    path_ = "outputs_0730_SB_HM125/two_jet/"

    # DNN with jet1_qgl jet2_qgl
    #path_ = "outputs_dnn_0628/two_jet/"
    #path_ = "outputs_dnn_0628_SB_HM125/two_jet/"
    # DNN without jet1_qgl jet2_qgl
    #path_ = "outputs_dnn_0730/two_jet/"
    #path_ = "outputs_dnn_0730_SB_HM125/two_jet/"
    categories = {
        "qqH_hmm": ["VBFHToMuMu_M125.root"],
        "ggH_hmm": ["GluGluHToMuMu_M125.root"],
        "DY": ["DY_105To160.root"],
        #"DY01J": ["DY_105To160_gen01J.root"],
        #"DY2J": ["DY_105To160_genOE2J.root"],
        "EWKZ": ["EWK_LLJJ_M105To160.root"],
        #"DY": ["DY_50FxFx.root"],
        #"DY": ["DY_50MLM.root"],
        #"EWKZ": ["EWK_LLJJ_M50.root"],
        "Top": ["ST_s-channel.root", "ST_t-channel_antitop.root", "ST_t-channel_top.root", "ST_tW_antitop.root", "ST_tW_top.root", "TTTo2L2Nu.root", "TTToSemiLep.root", "tZq.root"],
        "VV": ["ZZTo2L2Q.root","ZZTo2L2Nu.root","ZZTo4L.root","WZTo3LNu.root","WZTo2L2Q.root","WWTo2L2Nu.root"],
        "data_obs": ["data.root"],
    }

    histograms = {}
    for category, files in categories.items():
        full_paths = [path_ + file for file in files]
        # wrong input sig
        #if category == "VBFHmm":
        #    histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}", bin_edges, "eventWeight*6.373435026631877e-06/9.398e-05")
        #else:
        #    histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}", bin_edges, "eventWeight")
        # SR: outputs_0624/two_jet/
        #histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}", bin_edges, "eventWeight")
        # SB: outputs_0624_SB_HM125/two_jet/
        histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [125, 125], "bdt_score_t", f"{category}", bin_edges, "eventWeight")
        print("--------------------------")

    path_sys = "outputs_0731_2022EE_SB_HM125/two_jet/"
    categories_sys = {
        "qqH_hmm_sys": ["VBFHToMuMu_M125.root"],
    }
    for category, files in categories_sys.items():
        full_paths_sys = [path_sys + file for file in files]
        #histograms[category] = create_histogram(full_paths_sys, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}", bin_edges, "eventWeight")
        histograms[category] = create_histogram(full_paths_sys, "test", "diMufsr_rc_mass", [125, 125], "bdt_score_t", f"{category}", bin_edges, "eventWeight")

    histograms["qqH_hmm_sysUp"] = histograms["qqH_hmm_sys"].Clone("qqH_hmm_sysUp")
    histograms["qqH_hmm_sysUp"].Add(histograms["qqH_hmm"], -1.0)
    for bin_idx in range(1, histograms["qqH_hmm_sysUp"].GetNbinsX() + 1):
        bin_content = histograms["qqH_hmm_sysUp"].GetBinContent(bin_idx)
        histograms["qqH_hmm_sysUp"].SetBinContent(bin_idx, abs(bin_content))
    histograms["qqH_hmm_sysDown"] = histograms["qqH_hmm_sysUp"].Clone("qqH_hmm_sysDown")

    #filtered_histograms = (hist for key, hist in histograms.items() if key != 'qqH_hmm_sys')


    output_file = ROOT.TFile("vbf_ch3_vbfHmm_DY105To160_bdt_SB_0730noqgl_v3.root", "RECREATE")
    for key, hist in histograms.items():
        if key == 'qqH_hmm_sys':
            continue  # Skip the 'sys' histogram
        hist.Write()
    #for hist in histograms.values():
    #for hist in filtered_histograms.values():
    #    hist.Write()
    #output_file.Close()

if __name__ == "__main__":
    main()

