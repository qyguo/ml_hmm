#!/usr/bin/env python3
import ROOT, sys, os, math, traceback
from array import array
ROOT.gROOT.SetBatch(True)

# -------- user config --------
#GGH_FILE   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/GluGluHToMuMu_M125.root"
#QQH_FILE   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/VBFHToMuMu_M125.root"
GGH_FILE = "/eos/user/q/qguo/vbfhmm/ml/2024_v2/skimmed_ntuples/bdt_1111_2024/ggH/GluGluHToMuMu_M125.root"
QQH_FILE = "/eos/user/q/qguo/vbfhmm/ml/2024_v2/skimmed_ntuples/bdt_1111_2024/ggH/VBFHToMuMu_M125.root"
TREE_NAME  = "test"
MASS_NAME  = "diMufsr_rc_mass"      # must match branch name
#BDT_NAME   = "bdt_score_t"          # must match branch name
BDT_NAME  = "bdt_score"
WEIGHT_NAME = "eventWeight"         # set to "" if no weights, auto-checked below
outdir="datacards_v2_2024_bin3"

# 5 edges -> 4 cats (0..3). If you add the top 1.0 edge, it will still work.
# 2223 bdt_score_t
BDT_EDGES  = [0.0, 0.164179, 0.527363, 0.80597, 0.960199, 1.0]
# 2024
BDT_EDGES = [0.0, 0.402555, 0.572305, 0.742056, 0.877857, 1.0 ]  
BDT_EDGES = [0.0, 0.402555, 0.572305, 1.0 ]  



MASS_RANGE = (110., 150.)
NBINS      = 800

MH_RANGE   = (120., 130.)
MH_VALUE   = 125.0
# -----------------------------

def env():
    print("ROOT:", ROOT.gROOT.GetVersion())
    rc = ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
    print("Load libHiggsAnalysisCombinedLimit.so =", rc, "(0=ok)")

def open_file(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    return f

def import_safe(w, obj):
    imp = getattr(w, "import")
    try:
        imp(obj, ROOT.RooFit.RecycleConflictNodes(True))
    except Exception:
        imp(obj)

def const_spline(name, xvar, y):
    lo, hi = xvar.getMin(), xvar.getMax()
    xs = array('d', [lo, 0.5*(lo+hi), hi])
    ys = array('d', [y,  y,              y ])
    spl = ROOT.RooSpline1D(name, name, xvar, len(xs), xs, ys, "CSPLINE")
    spl._xarr = xs; spl._yarr = ys  # keep arrays alive
    return spl

def fit_dcb(dh, mh_like, tag):
    # tighter, sane ranges
    mean  = ROOT.RooRealVar(f"{tag}_mean",  f"{tag}_mean", 125.0, 120.0, 130.0)
    sigma = ROOT.RooRealVar(f"{tag}_sigma", f"{tag}_sigma",  2.0,   0.3,   5.0)
    aL    = ROOT.RooRealVar(f"{tag}_aL",    f"{tag}_aL",     1.2,   0.3,   5.0)
    nL    = ROOT.RooRealVar(f"{tag}_nL",    f"{tag}_nL",     4.0,   1.1,  20.0)
    aR    = ROOT.RooRealVar(f"{tag}_aR",    f"{tag}_aR",     1.3,   0.3,   5.0)
    nR    = ROOT.RooRealVar(f"{tag}_nR",    f"{tag}_nR",    10.0,   1.1,  50.0)

    pdf   = ROOT.RooDoubleCBFast(f"{tag}_fitpdf", f"{tag}_fitpdf",
                                 mh_like, mean, sigma, aL, nL, aR, nR)
    res = pdf.fitTo(dh, ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
    if not res or res.status() not in (0,1):
        print(f"[warn] {tag}: fit failed or bad status; using seeds.")
    else:
        print(f"[fit] {tag}: status={res.status()} mean={mean.getVal():.3f} sigma={sigma.getVal():.3f}")

    return {
        "peak": mean.getVal(), "sigma": sigma.getVal(),
        "aL": aL.getVal(), "nL": nL.getVal(),
        "aR": aR.getVal(), "nR": nR.getVal(),
        "norm": float(dh.sum(False))
    }

def build_one_proc_cat(w, proc, cat_idx, mh_wsp, MH, np_peak, np_sigma, fitvals):
    stem = f"{proc}_cat{cat_idx}_ggh"
    # constant splines seeded from the fit at MH=125
    spl_peak  = const_spline(f"{stem}_spline_peak",  MH, fitvals["peak"])
    spl_sigma = const_spline(f"{stem}_spline_sigma", MH, fitvals["sigma"])
    spl_aL    = const_spline(f"{stem}_spline_aL",    MH, fitvals["aL"])
    spl_nL    = const_spline(f"{stem}_spline_nL",    MH, fitvals["nL"])
    spl_aR    = const_spline(f"{stem}_spline_aR",    MH, fitvals["aR"])
    spl_nR    = const_spline(f"{stem}_spline_nR",    MH, fitvals["nR"])
    spl_norm  = const_spline(f"{stem}_spline_norm",  MH, fitvals["norm"])
    for s in [spl_peak, spl_sigma, spl_aL, spl_nL, spl_aR, spl_nR, spl_norm]:
        import_safe(w, s)

    # nuisance-shifted mean/sigma
    fpeak  = ROOT.RooFormulaVar(f"{stem}_fpeak",  "x[0]*(1+x[1])", ROOT.RooArgList(spl_peak,  np_peak))
    fsigma = ROOT.RooFormulaVar(f"{stem}_fsigma", "x[0]*(1+x[1])", ROOT.RooArgList(spl_sigma, np_sigma))
    import_safe(w, fpeak); import_safe(w, fsigma)

    # signal PDF
    pdf = ROOT.RooDoubleCBFast(f"{stem}_pdf", f"{stem}_pdf",
                               mh_wsp, fpeak, fsigma, spl_aL, spl_nL, spl_aR, spl_nR)
    import_safe(w, pdf)

    # norm helper (you can wrap into RooExtendPdf elsewhere if desired)
    pdf_norm = ROOT.RooFormulaVar(f"{stem}_pdf_norm", "@0", ROOT.RooArgList(spl_norm))
    import_safe(w, pdf_norm)

def build_dataset(ttree, lo, hi, cat_idx, proc, mh_src, bdt, weight_name):
    """Create weighted RooDataSet (via Import/Cut/WeightVar) and a 1D RooDataHist in mh_src."""
    # branch checks
    for nm in (mh_src.GetName(), bdt.GetName()):
        if not ttree.GetListOfBranches().FindObject(nm):
            raise RuntimeError(f"Tree missing branch '{nm}'")

    # vars (keep Python refs alive!)
    vars = ROOT.RooArgSet(); vars.add(mh_src); vars.add(bdt)

    # cut
    cut = f"{bdt.GetName()}>={lo} && {bdt.GetName()}<{hi}"

    # weight (optional)
    has_w = bool(weight_name and ttree.GetListOfBranches().FindObject(weight_name))
    if has_w:
        # RooFit.WeightVar expects a RooRealVar in the ArgSet to be present; define a dummy local var to register the name
        wvar = ROOT.RooRealVar(weight_name, weight_name, 1.0, -1e9, 1e9)
        vars.add(wvar)
        ds = ROOT.RooDataSet(
            f"data_all_{proc}_Tag{cat_idx}",
            f"data_all_{proc}_Tag{cat_idx}",
            vars,
            ROOT.RooFit.Import(ttree),
            ROOT.RooFit.Cut(cut),
            ROOT.RooFit.WeightVar(weight_name)
        )
    else:
        ds = ROOT.RooDataSet(
            f"data_all_{proc}_Tag{cat_idx}",
            f"data_all_{proc}_Tag{cat_idx}",
            vars,
            ROOT.RooFit.Import(ttree),
            ROOT.RooFit.Cut(cut)
        )

    if ds.numEntries() == 0 or ds.sumEntries() <= 0.0:
        print(f"[warn] {proc} cat{cat_idx}: empty after cut; entries={ds.numEntries()} sumW={ds.sumEntries():.3g}")

    # project to 1D datahist in the mass variable (weighted automatically)
    dh = ROOT.RooDataHist(
        f"data_{proc}_cat{cat_idx}_ggh_m125",
        f"data_{proc}_cat{cat_idx}_ggh_m125",
        ROOT.RooArgList(mh_src),
        ds
    )
    return ds, dh

def main():
    env()
    f_ggh = open_file(GGH_FILE)
    f_qqh = open_file(QQH_FILE)
    t_ggh = f_ggh.Get(TREE_NAME);  t_qqh = f_qqh.Get(TREE_NAME)
    if not t_ggh or not t_qqh:
        raise RuntimeError("TTree 'test' not found in one of the inputs")

    # reader variables that MATCH the TTree branches (not imported to workspace)
    mh_src = ROOT.RooRealVar(MASS_NAME, MASS_NAME, 0.5*(MASS_RANGE[0]+MASS_RANGE[1]), *MASS_RANGE); mh_src.setBins(NBINS)
    bdt    = ROOT.RooRealVar(BDT_NAME,  BDT_NAME, 0.5, 0.0, 1.0)

    # verify fundamental branches
    for nm in (MASS_NAME, BDT_NAME):
        if not t_ggh.GetListOfBranches().FindObject(nm):
            raise RuntimeError(f"ggH tree missing branch '{nm}'")
        if not t_qqh.GetListOfBranches().FindObject(nm):
            raise RuntimeError(f"qqH tree missing branch '{nm}'")

    # weight availability (once)
    has_weight_ggh = bool(WEIGHT_NAME and t_ggh.GetListOfBranches().FindObject(WEIGHT_NAME))
    has_weight_qqh = bool(WEIGHT_NAME and t_qqh.GetListOfBranches().FindObject(WEIGHT_NAME))
    if WEIGHT_NAME:
        print(f"[info] weight branch '{WEIGHT_NAME}': ggH={has_weight_ggh}  qqH={has_weight_qqh}")

    # iterate categories
    ncat = len(BDT_EDGES) - 1
    for i in range(ncat):
        lo, hi = BDT_EDGES[i], BDT_EDGES[i+1]
        print(f"\n=== Category {i}: BDT in [{lo}, {hi}) ===")

        # fresh workspace per category
        w = ROOT.RooWorkspace("w", "w")

        # workspace obs
        MH     = ROOT.RooRealVar("MH", "MH", MH_VALUE, *MH_RANGE)
        mh_wsp = ROOT.RooRealVar("mh_ggh","mh_ggh", 0.5*(MASS_RANGE[0]+MASS_RANGE[1]), *MASS_RANGE); mh_wsp.setBins(NBINS)
        import_safe(w, MH); import_safe(w, mh_wsp)

        # nuisances per category
        np_peak  = ROOT.RooRealVar(f"CMS_hmm_peak_cat{i}_ggh",  f"CMS_hmm_peak_cat{i}_ggh",  0.0, -1.0, 1.0)
        np_sigma = ROOT.RooRealVar(f"CMS_hmm_sigma_cat{i}_ggh", f"CMS_hmm_sigma_cat{i}_ggh", 0.0, -1.0, 1.0)
        import_safe(w, np_peak); import_safe(w, np_sigma)

        # datasets (weighted if branch exists)
        _, dh_ggh = build_dataset(t_ggh, lo, hi, i, "ggH", mh_src, bdt, WEIGHT_NAME if has_weight_ggh else "")
        _, dh_qqh = build_dataset(t_qqh, lo, hi, i, "qqH", mh_src, bdt, WEIGHT_NAME if has_weight_qqh else "")

        # import datahists, renaming mass to mh_ggh inside the workspace
        w.Import(dh_ggh, ROOT.RooFit.RenameVariable(MASS_NAME, "mh_ggh"))
        w.Import(dh_qqh, ROOT.RooFit.RenameVariable(MASS_NAME, "mh_ggh"))

        # quick fits in reader space to seed splines
        vals_ggh = fit_dcb(dh_ggh, mh_src, f"ggH_cat{i}_ggh")
        vals_qqh = fit_dcb(dh_qqh, mh_src, f"qqH_cat{i}_ggh")

        # build splines + PDFs in workspace using mh_ggh
        build_one_proc_cat(w, "ggH", i, mh_wsp, MH, np_peak, np_sigma, vals_ggh)
        build_one_proc_cat(w, "qqH", i, mh_wsp, MH, np_peak, np_sigma, vals_qqh)

        # write per-category file
        #outname = f"workspace_signal_cat{i}_ggh.root"
        outname = f"{outdir}/workspace_sig_cat{i}_ggh.root"
        fout = ROOT.TFile(outname, "RECREATE")
        w.Write()
        fout.Close()
        print(f"[ok] wrote {outname}")

if __name__ == "__main__":
    try:
        ROOT.EnableImplicitMT(False)
        ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2","migrad")
        ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(0)
        main()
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(1)
