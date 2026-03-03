import os
import sys
import ROOT
import array
from tqdm import tqdm

def create_histogram(file_paths, tree_name, var_to_cut, cut_range, var_to_plot,
                     hist_name, bin_edges, Weights):
    bin_array = array.array('d', bin_edges)
    hist = ROOT.TH1F(hist_name, f"Distribution of {var_to_plot}",
                     len(bin_edges)-1, bin_array)
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

        # cut string (weights * selection)
        cut_str = f"{Weights}*({var_to_cut} >= {cut_range[0]} && {var_to_cut} <= {cut_range[1]})"

        temp_hist_name = f"temp_{hist_name}"
        temp_hist = ROOT.TH1F(temp_hist_name, "", len(bin_edges)-1, bin_array)
        temp_hist.Sumw2()

        draw_cmd = f"{var_to_plot} >> {temp_hist_name}"
        tree.Draw(draw_cmd, cut_str, "goff")
        hist.Add(temp_hist)

        file.Close()
        print(f"Integral after adding from {file_path}: {hist.Integral()}")

    return hist


def get_mass_window(SR_or_SB):
    """Return [low, high] for diMufsr_kit_BSC_mass."""
    if SR_or_SB.upper() == "SR":
        return [115, 135]
    elif SR_or_SB.upper() == "SB":
        # your sideband-as-fixed-mass choice
        return [125, 125]
    else:
        raise ValueError(f"SR_or_SB must be 'SR' or 'SB', got {SR_or_SB}")


def main(SR_or_SB="SR"):
    # ---------------- user switch ----------------
    #SR_or_SB = "SB"   # <-- change to "SR" when needed


    # ---------------- binning (2024) ----------------
    bin_edges = [0.0, 0.28, 0.34, 0.4, 0.44, 0.51, 0.56, 0.62, 0.65, 0.69, 0.72, 0.74, 0.77, 0.8, 0.88, 1.0]
    # 2024_v4
    bin_edges = [0.0, 0.29, 0.36, 0.42, 0.47, 0.53, 0.59, 0.63, 0.67, 0.71, 0.75, 0.79, 0.83, 0.9, 1.0]
    bin_edges = [0.0, 0.29, 0.36, 0.42, 0.47, 0.5, 0.53, 0.59, 0.63, 0.67, 0.71, 0.75, 0.77, 0.79, 0.83, 0.9, 1.0]

    # ---------------- paths (2024) ----------------
    #base_path_2024 = "/eos/user/q/qguo/vbfhmm/ml/2024_v2/skimmed_ntuples/vbf_2024/"
    #tag = "_dnn_1123_2024"  # or whatever tag you want
    base_path_2024 = "/eos/user/z/zhangxu/sharing/hmm/2024_v4/skimmed_ntuples/SRSB/"
    tag = "_dnn_0203_2024"  # or whatever tag you want
    path_ = os.path.join(base_path_2024, f"{tag}{'_SB_HM125' if SR_or_SB.upper()=='SB' else ''}", "two_jet") + "/"

    print("Using path:", path_)
    print("Region:", SR_or_SB)

    categories = {
        "qqH_hmm": ["VBFHToMuMu_M125.root"],
        "ggH_hmm": ["GluGluHToMuMu_M125.root"],
        #"DY": ["DY_105To160_reweighted.root"],
        "DY": ["DY_105To160_ZpT-reweighted.root"],
        "EWKZ": ["EWK_LLJJ_M105To160.root"],
        "Top": ["ST_tW_antitop.root", "ST_tW_top.root", "TTTo2L2Nu.root"],
        "VV": ["ZZTo2L2Q.root","ZZTo2L2Nu.root","ZZTo4L.root","WZTo3LNu.root","WZTo2L2Q.root","WWTo2L2Nu.root"],
        "data_obs": ["data.root"],
    }

    mass_window = get_mass_window(SR_or_SB)

    histograms = {}
    for category, files in categories.items():
        full_paths = [os.path.join(path_, f) for f in files]

        histograms[category] = create_histogram(
            full_paths,
            #tree_name="test",
            tree_name="data_two_jet_m110To150",
            var_to_cut="diMufsr_kit_BSC_mass",
            cut_range=mass_window,
            var_to_plot="bdt_score_t",
            hist_name=category,
            bin_edges=bin_edges,
            Weights="eventWeight",
        )
        print("--------------------------")

    #outname = f"vbf_ch3_vbfHmm_bdt_t_{SR_or_SB}_2024_v2.root"
    # 20260204 2024_v4
    outname = f"vbf_ch3_vbfHmm_bdt_t_{SR_or_SB}_2024_v4_16Bin.root"
    output_file = ROOT.TFile(outname, "RECREATE")
    for hist in histograms.values():
        hist.Write()
    output_file.Close()
    print("Wrote:", outname)


if __name__ == "__main__":
    # Parse command line argument if given
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
