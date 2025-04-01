import ROOT
import array

def create_histogram(file_paths, tree_name, var_to_cut, cut_range, var_to_plot, hist_name, bin_edges):
    # Create a histogram with variable bin sizes using array
    bin_array = array.array('d', bin_edges)
    hist = ROOT.TH1F(hist_name, f"Distribution of {var_to_plot}", len(bin_edges)-1, bin_array)

    # Process each file
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
        
        cut_str = f"{var_to_cut} >= {cut_range[0]} && {var_to_cut} <= {cut_range[1]}"
        draw_cmd = f"{var_to_plot} >> {hist_name}"
        print(f"Draw command: {draw_cmd}, Cut: {cut_str}")

        tree.Draw(draw_cmd, cut_str, "goff")
        print(f"Integral after drawing from {file_path}: {hist.Integral()}")

        file.Close()

    return hist

def main():
    # Define bin edges and the path prefix
    bin_edges = [0.0, 0.06, 0.12, 0.2, 0.3, 0.41, 0.49, 0.58, 0.64, 0.78, 1.0]
    path_ = "outputs_0617/two_jet/"
    
    # Define categories and associated files
    categories = {
        "VBFHmm": ["VBFHToMuMu_M125.root"],
        "DY": ["DY_105To160.root"],
        "EWK": ["EWK_LLJJ_M105To160.root"],
        "Top": ["ST_s-channel.root", "ST_t-channel_antitop.root", "ST_t-channel_top.root", "ST_tW_antitop.root", "ST_tW_top.root", "TTTo2L2Nu.root", "TTToSemiLep.root", "tZq.root"],
    }
    
    # Create histograms for each category
    histograms = {}
    for category, files in categories.items():
        full_paths = [path_ + file for file in files]
        print("-----------------------------------------")
        histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [0, 100000], "bdt_score_t", f"{category}", bin_edges)

    # Save the histograms to a ROOT file
    output_file = ROOT.TFile("output_hist_forCombine.root", "RECREATE")
    for hist in histograms.values():
        hist.Write()
    output_file.Close()

if __name__ == "__main__":
    main()

