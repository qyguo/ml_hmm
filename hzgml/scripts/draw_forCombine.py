import ROOT
import array

# Function to create a histogram for each category
def create_histogram(file_paths, tree_name, var_to_cut, cut_range, var_to_plot, hist_name, bins, range):
    # Create a new histogram
    hist = ROOT.TH1F(hist_name, hist_name, bins, range[0], range[1])
    #h_bkgmc_cen = TH1F('h_bkgmc_cen', 'h_bkgmc_cen', nscan, 0., 1.)

    # Process each file
    for file_path in file_paths:
        # Open the ROOT file
        file = ROOT.TFile.Open(file_path)
        tree = file.Get(tree_name)
        print(file_path, tree.GetEntries())
        
        # Define the cut and the variable to plot
        cut = f"eventWeight* ( {var_to_cut} >= {cut_range[0]} && {var_to_cut} <= {cut_range[1]} )"
        draw_cmd = f"{var_to_plot} >>+ {hist_name}"

        # Draw the variable into the histogram with the cut
        tree.Draw(draw_cmd, cut, "goff")

        # Close the file
        file.Close()
    
    return hist

# Function to create a histogram with variable bin sizes for each category
def create_histogram(file_paths, tree_name, var_to_cut, cut_range, var_to_plot, hist_name, bin_edges):
    # Convert Python list of bin edges to a ROOT array
    #bin_array = ROOT.std.vector('double')(len(bin_edges))
    #bin_array = ROOT.std.vector('double')()
    #for edge in bin_edges:
    #    bin_array.push_back(edge)
    bin_array = array.array('d', bin_edges)  # 'd' is the type code for double
    print("nbins", len(bin_edges)-1, bin_array)
    print("nbins", len(bin_edges)-1, bin_edges)

    # Create the histogram with variable bin sizes
    #hist = ROOT.TH1F(hist_name, hist_name, len(bin_edges)-1, bin_array.data())
    hist = ROOT.TH1F(hist_name, hist_name, len(bin_edges)-1, bin_array)
    #hist = ROOT.TH1F('hist', 'hist', 10, 0., 1.)
    # Process each file
    for file_path in file_paths:

        print("Opening file:", file_path)  # Debug: print the file path
        # Open the ROOT file
        file = ROOT.TFile.Open(file_path)
        if file.IsOpen():
            print("File opened successfully.")
        else:
            print("Failed to open file.")
            continue

        tree = file.Get(tree_name)
        if tree:
            print(f"Tree '{tree_name}' found with {tree.GetEntries()} entries.")
        else:
            print(f"Tree '{tree_name}' not found.")
            file.Close()
            continue


        # Open the ROOT file
        #file = ROOT.TFile.Open(file_path)
        #tree = file.Get(tree_name)
        print(file_path, tree.GetEntries())

        # Define the cut and the variable to plot
        cut = f"{var_to_cut} >= {cut_range[0]} && {var_to_cut} <= {cut_range[1]}"
        draw_cmd = f"{var_to_plot} >> {hist_name}"
        print(f"Draw command: {draw_cmd}, Cut: {cut}")

        # Draw the variable into the histogram with the cut
        #tree.Draw(draw_cmd, cut, "goff")
        tree.Draw(draw_cmd, cut)
        print("Integral: ", file_path, hist.Integral())

        # Close the file
        file.Close()

    return hist

# Main function to process the files and create histograms
def main():
    # Define the histogram settings
    #bins = 50
    #range_bdt_score = (0, 1)
    bin_edges = [0.0, 0.06, 0.12, 0.2, 0.3, 0.41, 0.49, 0.58, 0.64, 0.78, 1.0]
    path_ = "outputs_0617/two_jet/"
    
    # Files and categories
    categories = {
        "VBFHmm": ["VBFHToMuMu_M125.root"],
        "DY": ["DY_105To160.root"],
        "EWK": ["EWK_LLJJ_M105To160.root"],
        "Top": ["ST_s-channel.root","ST_t-channel_antitop.root","ST_t-channel_top.root","ST_tW_antitop.root","ST_tW_top.root","TTTo2L2Nu.root","TTToSemiLep.root","tZq.root"],
        #"Others": []  # Assuming you have additional files for 'Others'
    }
    
    # Create histograms for each category
    histograms = {}
    for category, files in categories.items():
        full_paths = [path_ + file for file in files]
        #histograms[category] = create_histogram(files, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}_bdt_score", bins, range_bdt_score)
        #histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [115, 135], "bdt_score_t", f"{category}", bin_edges)
        histograms[category] = create_histogram(full_paths, "test", "diMufsr_rc_mass", [0, 100000], "bdt_score_t", f"{category}", bin_edges)

    # Optionally, save the histograms to a new ROOT file
    output_file = ROOT.TFile("output_hist_forCombine.root", "RECREATE")
    for hist in histograms.values():
        hist.Write()
    output_file.Close()

# Run the main function
if __name__ == "__main__":
    main()

