#!/usr/bin/env python3
import ROOT, sys, os, platform, math, traceback
from array import array
ROOT.gROOT.SetBatch(True)

# ---------------- user config ----------------
#INPUT_FILE = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/data.root"
INPUT_FILE = "/eos/user/q/qguo/vbfhmm/ml/2024_v2/skimmed_ntuples/bdt_1111_2024/ggH/data_ggh_vbf.root"
TREE_NAME  = "test"
MASS_NAME  = "diMufsr_rc_mass"
#BDT_NAME   = "bdt_score_t"
BDT_NAME   = "bdt_score"
WEIGHT     = ""                 # empty -> unweighted
MASS_RANGE = (110., 150.)
NBINS      = 800
# 2223 bdt_score_t
#BDT_EDGES  = [0.0,0.164179,0.527363,0.80597,0.960199,1.0]   # 5 categories -> 0..4
BDT_EDGES  = [0.0, 0.402555, 0.572305, 0.742056, 0.877857 ,1.0]
BDT_EDGES  = [0.0, 0.402555, 0.572305, 1.0]
# ---------------------------------------------
outdir="datacards_v2_2024_bin3"

def set_integrator_defaults():
    cfg = ROOT.RooAbsReal.defaultIntegratorConfig()
    cfg.setEpsAbs(1e-7)
    cfg.setEpsRel(1e-7)
    try:
        cfg.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")
    except Exception as e:
        print("[warn] keep default 1D integrator:", e)

def env_banner():
    print("=== Basic environment ===")
    print("Python :", sys.version.replace("\n"," "))
    print("Platform:", platform.platform())
    print("ROOT    :", ROOT.gROOT.GetVersion(), " | PyROOT:", "cppyy" if hasattr(ROOT,"cppyy") else "unknown")
    print("\n=== Loading Combine lib ===")
    rc = ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
    print("Load libHiggsAnalysisCombinedLimit.so:", rc, " (0=ok)")
    print("-----------------------------------------------------\n")

def is_bad_number(x):
    return (x is None) or (not math.isfinite(x))

def safe_integral(tag, pdf, mh, rng="full"):
    try:
        mh.setRange(rng, mh.getMin(), mh.getMax())
        integ = pdf.createIntegral(ROOT.RooArgSet(mh), ROOT.RooFit.Range(rng))
        val = float(integ.getVal(ROOT.RooArgSet(mh)))
    except Exception:
        val = float("nan")
    print(f"[dbg] {tag}: integral('{rng}') = {val:.6g}")
    return val

def sane_pdf(tag, pdf, mh, max_ok=1e6):
    val = safe_integral(tag, pdf, mh, "full")
    if is_bad_number(val) or val <= 0.0:
        print(f"[skip] {tag}: integral non-positive or NaN -> skip this candidate.")
        return False
    if val > max_ok:
        print(f"[skip] {tag}: integral {val:.3g} is too large (> {max_ok:g}) -> skip.")
        return False
    return True

# ---------- helpers made tag-aware so names never collide ----------
def build_transfer_cheb3(mh, stem, cat_tag):
    v1 = ROOT.RooRealVar(f"{stem}_transfer_order3_coef_1_{cat_tag}", f"{stem}_transfer_order3_coef_1_{cat_tag}",  0.00, -10, 10)
    v2 = ROOT.RooRealVar(f"{stem}_transfer_order3_coef_2_{cat_tag}", f"{stem}_transfer_order3_coef_2_{cat_tag}",  0.00, -10, 10)
    v3 = ROOT.RooRealVar(f"{stem}_transfer_order3_coef_3_{cat_tag}", f"{stem}_transfer_order3_coef_3_{cat_tag}",  0.00, -10, 10)
    f1 = ROOT.RooFormulaVar(f"{stem}_transfer_order3_coef_1_{cat_tag}_sq", "x[0]", ROOT.RooArgList(v1))
    f2 = ROOT.RooFormulaVar(f"{stem}_transfer_order3_coef_2_{cat_tag}_sq", "x[0]", ROOT.RooArgList(v2))
    f3 = ROOT.RooFormulaVar(f"{stem}_transfer_order3_coef_3_{cat_tag}_sq", "x[0]", ROOT.RooArgList(v3))
    cheb = ROOT.RooChebychev(f"{stem}_transfer_order3_{cat_tag}_pdf",
                             f"{stem}_transfer_order3_{cat_tag}_pdf",
                             mh, ROOT.RooArgList(f1,f2,f3))
    return cheb, (v1,v2,v3,f1,f2,f3)

def make_roomodz(mh, cat_tag, cat_ggh):
    a = ROOT.RooRealVar(f"bwzr_{cat_ggh}_coef1", f"bwzr_{cat_ggh}_coef1",  0.00, -0.10, 0.10)
    b = ROOT.RooRealVar(f"bwzr_{cat_ggh}_coef2", f"bwzr_{cat_ggh}_coef2",  0.00, -0.10, 0.10)
    c = ROOT.RooRealVar(f"bwzr_{cat_ggh}_coef3", f"bwzr_{cat_ggh}_coef3",  0.00, -0,5)
    try:
        core = ROOT.RooModZPdf(f"bwzr_{cat_ggh}_pdf", f"bwzr_{cat_ggh}_pdf", mh, a, b, c)
        return core, (a,b,c)
    except Exception as e:
        print("[warn] RooModZPdf failed; fallback Chebychev(3).", e)
        core = ROOT.RooChebychev(f"bwzr_{cat_ggh}_pdf", f"bwzr_{cat_ggh}_pdf", mh, ROOT.RooArgList(a,b,c))
        return core, (a,b,c)

def make_sum2exp(mh, cat_tag, cat_ggh):
    e1   = ROOT.RooRealVar(f"exp_order2_{cat_ggh}_coef1", f"exp_order2_{cat_ggh}_coef1", -0.02,  -1.0, -1e-5)
    e2   = ROOT.RooRealVar(f"exp_order2_{cat_ggh}_coef2", f"exp_order2_{cat_ggh}_coef2", -0.005, -1.0, -1e-5)
    frac = ROOT.RooRealVar(f"exp_order2_{cat_ggh}_frac1", f"exp_order2_{cat_ggh}_frac1",  0.20,    0.,   1.00)
    core = ROOT.RooSumTwoExpPdf(f"exp_{cat_ggh}_pdf", f"exp_{cat_ggh}_pdf", mh, e1, e2, frac)
    return core, (e1,e2,frac)

def make_fewz_spline(mh, cat_tag, cat_ggh):
    lo, hi = mh.getMin(), mh.getMax()
    xarr = array('d', [lo, 0.5*(lo+hi), hi])
    yarr = array('d', [0.03, 0.04, 0.05])
    npts = len(xarr)
    spl = ROOT.RooSpline1D(f"fewz_1j_spl_order1_{cat_ggh}",
                           f"fewz_1j_spl_order1_{cat_ggh}",
                           mh, npts, xarr, yarr, "CSPLINE")
    spl._xarr = xarr; spl._yarr = yarr   # keep alive
    gen = ROOT.RooGenericPdf(f"fewz_1j_spl_pdf", "x[0]", ROOT.RooArgList(spl))
    return gen, spl

def fit_and_get_minNLL(tag, pdf, data, rng="full"):
    mh = data.get().first()
    if not sane_pdf(tag, pdf, mh):
        return None, None
    res = pdf.fitTo(
        data,
        ROOT.RooFit.Range(rng),
        ROOT.RooFit.NumCPU(1),
        ROOT.RooFit.Offset(True),
        ROOT.RooFit.Minimizer("Minuit2","migrad"),
        ROOT.RooFit.Save(True),
        ROOT.RooFit.PrintLevel(1),
        ROOT.RooFit.Warnings(True),
        ROOT.RooFit.SumW2Error(True),
    )
    if not res:
        print(f"[warn] {tag}: fitTo returned None")
        return None, None
    mnll = float(res.minNll()) if hasattr(res, "minNll") else None
    print(f"[fit] {tag} done; status={res.status()}, edm={res.edm():.3g}, minNLL={mnll}")
    return res, mnll
# ------------------------------------------------------------------

def build_one_category(cat_idx, lo_edge, hi_edge):
    cat_tag = f"cat{cat_idx}_ggh"
    cat_ggh = f"cat_ggh"
    print(f"\nq=== Building {cat_tag}  (bdt in [{lo_edge},{hi_edge})) ===")

    f = ROOT.TFile.Open(INPUT_FILE)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {INPUT_FILE}")
    t = f.Get(TREE_NAME)
    if not t:
        raise RuntimeError(f"TTree '{TREE_NAME}' not found")

    # fresh workspace per category -> file-per-cat, no name conflicts
    w  = ROOT.RooWorkspace("w","w"); wimp = getattr(w,"import")

    mh  = ROOT.RooRealVar(MASS_NAME, MASS_NAME, 0.5*(MASS_RANGE[0]+MASS_RANGE[1]), *MASS_RANGE)
    bdt = ROOT.RooRealVar(BDT_NAME,  BDT_NAME, 0.5, 0.0, 1.0)
    mh.setBins(NBINS)

    cols = ROOT.RooArgSet(mh, bdt)
    args = [ROOT.RooFit.Import(t), ROOT.RooFit.Cut(f"{BDT_NAME}>={lo_edge} && {BDT_NAME}<{hi_edge}")]
    use_w = bool(WEIGHT and WEIGHT.strip() and t.GetListOfBranches().FindObject(WEIGHT))
    if use_w:
        args.append(ROOT.RooFit.WeightVar(WEIGHT))
    ds_all = ROOT.RooDataSet(f"data_all_Tag{cat_idx}", f"data_all_Tag{cat_idx}", cols, *args)

    # rename mh in this workspace to a consistent name
    dh = ROOT.RooDataHist(f"data_{cat_tag}", f"data_{cat_tag}", ROOT.RooArgList(mh), ds_all)
    wimp(dh, ROOT.RooFit.RenameVariable(MASS_NAME,"mh_ggh"))
    mhw = w.var("mh_ggh"); mhw.setRange("full", *MASS_RANGE)
    data = w.data(f"data_{cat_tag}")
    nevt = float(data.sum(False))
    print(f"[dbg] data entries (weight-sum) in {cat_tag}: {nevt:.3f}; nbins={mhw.getBins()}")

    # cores
    bwzr_core, bwzr_pars = make_roomodz(mhw, cat_tag, cat_ggh)
    exp_core,  exp_pars  = make_sum2exp(mhw, cat_tag, cat_ggh)
    fewz_pdf, fewz_spl   = make_fewz_spline(mhw, cat_tag, cat_ggh)

    # fewz modifier (Bernstein 3) — tag-aware
    bf3 = ROOT.RooBernsteinFast(3)
    b1 = ROOT.RooRealVar(f"fewz_1j_spl_order3_bern_{cat_ggh}_coef1", f"fewz_1j_spl_order3_bern_{cat_ggh}_coef1", 1.0, -10.0, 10.0)
    b2 = ROOT.RooRealVar(f"fewz_1j_spl_order3_bern_{cat_ggh}_coef2", f"fewz_1j_spl_order3_bern_{cat_ggh}_coef2", 1.0, -10.0, 10.0)
    b3 = ROOT.RooRealVar(f"fewz_1j_spl_order3_bern_{cat_ggh}_coef3", f"fewz_1j_spl_order3_bern_{cat_ggh}_coef3", 1.0, -10.0, 10.0)
    fewz_mod = bf3(f"fewz_1j_spl_order3_bern_{cat_ggh}_pdf", f"fewz_1j_spl_order3_bern_{cat_ggh}_pdf", mhw, ROOT.RooArgList(b1,b2,b3))

    # transfers
    bwzr_trf, (bw1,bw2,bw3,bw1sq,bw2sq,bw3sq) = build_transfer_cheb3(mhw, f"bwzr_{cat_ggh}_pdf", cat_tag)
    exp_trf,  (ex1,ex2,ex3,ex1sq,ex2sq,ex3sq) = build_transfer_cheb3(mhw, f"exp_{cat_ggh}_pdf",  cat_tag)
    fewz_trf, (fz1,fz2,fz3,fz1sq,fz2sq,fz3sq) = build_transfer_cheb3(mhw, f"fewz_1j_spl_{cat_ggh}_pdf", cat_tag)

    # products (for fewz we use 'fewz_mod'; swap with 'fewz_trf' if preferred)
    bwzr_prod = ROOT.RooProdPdf(f"bkg_bwzr_{cat_tag}_pdf", f"bkg_bwzr_{cat_tag}_pdf",
                                ROOT.RooArgList(bwzr_core, bwzr_trf))
    exp_prod  = ROOT.RooProdPdf(f"bkg_exp_{cat_tag}_pdf", f"bkg_exp_{cat_tag}_pdf",
                                ROOT.RooArgList(exp_core, exp_trf))
    fewz_prod = ROOT.RooProdPdf(f"bkg_fewz_1j_spl_{cat_tag}_pdf", f"bkg_fewz_1j_spl_{cat_tag}_pdf",
                                ROOT.RooArgList(fewz_pdf, fewz_mod))

    # fit each and select best by minNLL
    results = []
    for name, pdf in [("bwzr",bwzr_prod), ("exp",exp_prod), ("fewz",fewz_prod)]:
        res, mnll = fit_and_get_minNLL(f"{cat_tag}:{name}", pdf, data, "full")
        if res is not None and mnll is not None and res.status() in (0,1):
            results.append((name, pdf, mnll))
    if not results:
        raise RuntimeError(f"[FATAL] no successful fits in {cat_tag}")
    results.sort(key=lambda x: x[2])
    best_name, best_pdf, best_nll = results[0]
    print(f"[best] {cat_tag}:{best_name} with NLL={best_nll:.6g}")

    # MultiPdf (index per cat)
    #idx = ROOT.RooCategory(f"pdf_index_{cat_tag}", "pdf index")
    idx = ROOT.RooCategory("pdf_index_ggh","pdf index")
    pdfs = ROOT.RooArgList(); pdfs.add(bwzr_prod); pdfs.add(exp_prod); pdfs.add(fewz_prod)
    multipdf = ROOT.RooMultiPdf(f"bkg_{cat_tag}_pdf", f"bkg_{cat_tag}_pdf", idx, pdfs)
    idx.setIndex({"bwzr":0,"exp":1,"fewz":2}.get(best_name,0))

    # norms (per cat)
    bkg_pdf_norm  = ROOT.RooRealVar(f"bkg_{cat_tag}_pdf_norm", "bkg yield", float(nevt), -1e+30, 1e+30)
    core_pdf_norm = ROOT.RooRealVar(f"bkg_core_ggh_pdf_norm", "core yield", float(nevt), -1e+30, 1e+30)

    # import everything (recycling where possible)
    wimp(idx, ROOT.RooFit.RecycleConflictNodes(True))
    wimp(multipdf, ROOT.RooFit.RecycleConflictNodes(True))
    for obj in [
        bkg_pdf_norm, core_pdf_norm,
        bwzr_prod, exp_prod, fewz_prod,
        bwzr_core, exp_core, fewz_pdf,
        bwzr_trf,  exp_trf,  fewz_trf,
        fewz_mod, fewz_spl,
        *bwzr_pars, *exp_pars,
        b1,b2,b3,
        bw1,bw2,bw3,bw1sq,bw2sq,bw3sq,
        ex1,ex2,ex3,ex1sq,ex2sq,ex3sq,
        fz1,fz2,fz3,fz1sq,fz2sq,fz3sq,
    ]:
        try:
            wimp(obj, ROOT.RooFit.RecycleConflictNodes(True))
        except Exception as e:
            nm = obj.GetName() if hasattr(obj,"GetName") else str(obj)
            print(f"[imp-warn] failed to import {nm}: {e}")

    w.Print()
    out_name = f"{outdir}/workspace_bkg_cat{cat_idx}_ggh_ori3_2024.root"
    out = ROOT.TFile(out_name, "RECREATE")
    w.Write(); out.Close()
    print(f"[ok] wrote {out_name}")

def main():
    ROOT.EnableImplicitMT(False)
    ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2","migrad")
    ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(0)

    # common once
    env_banner()
    set_integrator_defaults()

    try:
        ncat = len(BDT_EDGES)-1
        for i in range(ncat):
            lo, hi = BDT_EDGES[i], BDT_EDGES[i+1]
            build_one_category(i, lo, hi)
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
