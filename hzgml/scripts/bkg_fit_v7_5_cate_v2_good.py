#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python3 bkg_fit_v7_5_cate_v2_good.py   --input /eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0717_bdt_0716_2223/ggH/data.root --tree test   --mass-branch diMufsr_rc_mass   --cat-var bdt_score_t   --cat-edges 0.0,0.079602,0.273632,0.776119,0.945274,1.0   --blind 120 130
# trained with sig=ggH
# python3 bkg_fit_v7_5_cate_v2_good.py --input /eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/data.root --tree test   --mass-branch diMufsr_rc_mass   --blind 120 130 --out bkg_fit_v7_out_v3_0820 --cat-var bdt_score_t   --cat-edges 0.0,0.164179,0.527363,0.80597,0.960199,1.0

import argparse, os
from array import array
import ROOT as R
R.gROOT.SetBatch(True)
R.RooMsgService.instance().setGlobalKillBelow(R.RooFit.WARNING)

# -----------------------------
# Helpers
# -----------------------------
def has_branch(t, name):
    brs = t.GetListOfBranches()
    return (brs and brs.FindObject(name) is not None)

def autodetect_weight(t):
    for nm in ("weight","evtWeight","eventWeight","w","puWeight"):
        if has_branch(t, nm): return nm
    return None

def fix_addpdf_norm(pdf, x):
    if isinstance(pdf, R.RooAddPdf):
        pdf.fixCoefNormalization(R.RooArgSet(x))

def free_params(pdf, data):
    pars = pdf.getParameters(data)
    it = pars.createIterator()
    n = 0
    obj = it.Next()
    while obj:
        if not obj.isConstant(): n += 1
        obj = it.Next()
    return n

def fit_unbinned(pdf, data, fit_range="fit", max_retries=1):
    nll = pdf.createNLL(
        data,
        R.RooFit.Range(fit_range),
        R.RooFit.Offset(True),
        R.RooFit.NumCPU(1),
    )
    m = R.RooMinimizer(nll)
    m.setPrintLevel(-1)
    m.setStrategy(1)
    m.optimizeConst(2)
    status = m.migrad()
    m.hesse()
    tries = 0
    while (status != 0) and (tries < max_retries):
        tries += 1
        m.setStrategy(2)
        status = m.migrad()
        m.hesse()
    res = m.save()
    res._keep = [nll]
    return res

def chi2_on_binned(pdf, data, x, nbins, fit_range="fit"):
    x.setBins(nbins, fit_range)
    binned = R.RooDataHist("binned","binned", R.RooArgList(x), data)
    chi2 = R.RooChi2Var("chi2","chi2", pdf, binned,
                        R.RooFit.Range(fit_range),
                        R.RooFit.DataError(R.RooAbsData.Auto))
    npar = free_params(pdf, data)
    ndf  = nbins - npar - 1
    return chi2.getVal(), max(ndf, 1)

# ---- ratio helpers (use RooCurve/TGraph) ----
def graph_ratio(num: R.TGraph, denom: R.TGraph, name, color, style, width=3):
    """Return a TGraph with y_i = num(x_i)/denom(x_i) using denom.Eval(x)."""
    n = num.GetN()
    gr = R.TGraph()
    gr.SetName(name)
    gr.SetLineColor(color)
    gr.SetLineStyle(style)
    gr.SetLineWidth(width)
    gr.SetMarkerStyle(0)
    xbuf = array('d',[0.0]); ybuf = array('d',[0.0])
    for i in range(n):
        num.GetPoint(i, xbuf, ybuf)
        x = float(xbuf[0]); y = float(ybuf[0])
        yd = max(denom.Eval(x), 1e-300)
        gr.SetPoint(gr.GetN(), x, y/yd)
    return gr

def find_curve_by_name(frame: R.RooPlot, name: str):
    for i in range(int(frame.numItems())):
        obj = frame.getObject(i)
        if obj and hasattr(obj, "GetName") and obj.GetName() == name:
            return obj
    return None

# -----------------------------
# PDF building blocks
# -----------------------------
def bernstein(nm, x, order=3):
    coefs, keep = R.RooArgList(), []
    for i in range(order+1):
        ci = R.RooRealVar(f"{nm}_c{i}", f"{nm}_c{i}", 0.2, 0.0, 50.0)
        keep.append(ci); coefs.add(ci)
    pdf = R.RooBernstein(nm, nm, x, coefs); pdf._keep = keep
    return pdf

def bwz_redux(name, x):
    mZ  = R.RooRealVar(f"{name}_mZ",  f"{name}_mZ", 91.1876, 88., 94.)
    gZ  = R.RooRealVar(f"{name}_gZ",  f"{name}_gZ", 2.4952,   1.5, 4.0)
    bw  = R.RooBreitWigner(f"{name}_bw", f"{name}_bw", x, mZ, gZ)
    p   = R.RooRealVar(f"{name}_p",   f"{name}_p",   -2.5, -20., -0.2)
    cont= R.RooGenericPdf(f"{name}_plaw", f"{name}_plaw", "pow(@0,@1)", R.RooArgList(x,p))
    fbw = R.RooRealVar(f"{name}_fbw", f"{name}_fbw", 0.25, 0.0, 1.0)
    pdf = R.RooAddPdf(name, name, bw, cont, fbw)
    fix_addpdf_norm(pdf, x)
    pdf._keep = [mZ,gZ,p,fbw,bw,cont]
    return pdf

def fewz_like(name, x):
    beta = R.RooRealVar(f"{name}_beta", f"{name}_beta", 3.0, 0.5, 8.0)
    c1   = R.RooRealVar(f"{name}_c1",   f"{name}_c1",   20.0, -500., 500.)
    c2   = R.RooRealVar(f"{name}_c2",   f"{name}_c2",   600.0, -2e4, 2e4)
    pdf  = R.RooGenericPdf(
        name, name,
        "( pow(@0,-@1) * (1 + @2/(@0+1e-12) + @3/((@0+1e-12)*(@0+1e-12))) ) + 1e-300",
        R.RooArgList(x,beta,c1,c2)
    )
    pdf._keep = [beta,c1,c2]
    return pdf

def prod(name, a, b):
    p = R.RooProdPdf(name, name, R.RooArgList(a,b))
    p._keep = [a,b]
    return p

def bwz_x_bern(name, x):
    core = bwz_redux(f"{name}_core", x)
    ber  = bernstein(f"{name}_bern", x, 3)
    return prod(name, core, ber)

def fewz_x_bern(name, x):
    core = fewz_like(f"{name}_core", x)
    ber  = bernstein(f"{name}_bern", x, 3)
    return prod(name, core, ber)

def bwzgamma(name, x):
    core = bwz_redux(f"{name}_bwz", x)
    k     = R.RooRealVar(f"{name}_k",     f"{name}_k",     6.0,  1.2, 20.0)
    theta = R.RooRealVar(f"{name}_theta", f"{name}_theta", 4.0,  0.5, 40.0)
    gam = R.RooGenericPdf(
        f"{name}_gamma", f"{name}_gamma",
        "TMath::Max( TMath::Power(@0, @1-1)*TMath::Exp(-@0/@2) / ( TMath::Gamma(@1)*TMath::Power(@2,@1) ), 1e-300 )",
        R.RooArgList(x,k,theta)
    )
    f    = R.RooRealVar(f"{name}_f", f"{name}_f", 0.30, 0.0, 1.0)
    pdf  = R.RooAddPdf(name, name, core, gam, f)
    fix_addpdf_norm(pdf, x)
    pdf._keep = [core,k,theta,gam,f]
    return pdf

def landau_x_bern(name, x):
    mp   = R.RooRealVar(f"{name}_mp",  f"{name}_mp",  70.,  40., 120.)
    sig  = R.RooRealVar(f"{name}_sig", f"{name}_sig", 12.,   2.,  40.)
    lan  = R.RooLandau(f"{name}_lan",  f"{name}_lan", x, mp, sig)
    ber  = bernstein(f"{name}_bern", x, 3)
    p    = prod(name, lan, ber)
    p._keep += [mp,sig,lan,ber]
    return p

def s_exponential(name, x, n=3):
    pdfs, fracs, keep = R.RooArgList(), R.RooArgList(), []
    for i in range(n):
        tau = R.RooRealVar(f"{name}_tau{i+1}", f"{name}_tau{i+1}", -(0.02+0.01*i), -1.0, -1e-4)
        ei  = R.RooExponential(f"{name}_e{i+1}", f"{name}_e{i+1}", x, tau)
        keep += [tau, ei]
        pdfs.add(ei)
        if i < n-1:
            f = R.RooRealVar(f"{name}_f{i+1}", f"{name}_f{i+1}", 1.0/n, 0.0, 1.0)
            keep.append(f); fracs.add(f)
    pdf = R.RooAddPdf(name, name, pdfs, fracs, True)
    fix_addpdf_norm(pdf, x)
    pdf._keep = keep
    return pdf

def s_powerlaw(name, x, n=3):
    pdfs, fracs, keep = R.RooArgList(), R.RooArgList(), []
    for i in range(n):
        p  = R.RooRealVar(f"{name}_p{i+1}", f"{name}_p{i+1}", -(1.8+0.4*i), -20., -0.2)
        pi = R.RooGenericPdf(f"{name}_pl{i+1}", f"{name}_pl{i+1}", "pow(@0,@1)", R.RooArgList(x,p))
        keep += [p, pi]
        pdfs.add(pi)
        if i < n-1:
            f = R.RooRealVar(f"{name}_f{i+1}", f"{name}_f{i+1}", 1.0/n, 0.0, 1.0)
            keep.append(f); fracs.add(f)
    pdf = R.RooAddPdf(name, name, pdfs, fracs, True)
    fix_addpdf_norm(pdf, x)
    pdf._keep = keep
    return pdf

# -----------------------------
# Fit + plot for one category
# -----------------------------
def do_category(cat_idx, lo, hi, full_data, m, cat, args):
    label = f"cat{cat_idx}_[{lo:.3g},{hi:.3g})"
    print(f"\n=== Category {cat_idx}: {cat.GetName()} in [{lo},{hi}) ===")
    cut = f"{cat.GetName()}>={lo} && {cat.GetName()}<{hi}"
    data = full_data.reduce(R.RooFit.Cut(cut))

    models = [
        ("BWZRedux",        bwz_redux("BWZRedux", m)),
        ("BWZxBernstein",   bwz_x_bern("BWZxBern", m)),
        ("S-Power-Law",     s_powerlaw("SPower", m, 3)),
        ("S-Exponential",   s_exponential("SExp", m, 3)),
        ("BWZGamma",        bwzgamma("BWZGamma", m)),
        ("FEWZxBernstein",  fewz_x_bern("FEWZxBern", m)),
        ("LandauxBernstein",landau_x_bern("LandauBern", m)),
    ]
    style = {
        "BWZRedux":        (R.kRed+1,       1),
        "BWZxBernstein":   (R.kBlue+1,      1),
        "S-Power-Law":     (R.kCyan+1,      2),
        "S-Exponential":   (R.kOrange+7,    7),
        "BWZGamma":        (R.kGreen+2,     3),
        "FEWZxBernstein":  (R.kMagenta+1,   9),
        "LandauxBernstein":(R.kGray+2,      3),
    }

    # Fit & chi2
    results, chi2_table = {}, []
    for name, pdf in models:
        print(f"[Fit] {name}")
        res = fit_unbinned(pdf, data, "fit", max_retries=1)
        results[name] = res
        chi2, ndf = chi2_on_binned(pdf, data, m, args.bins, "fit")
        chi2_table.append((name, chi2/ndf, ndf))

    # Frame & data (with optional blinding)
    #frame = m.frame(R.RooFit.Range("fit"))
    frame = m.frame(R.RooFit.Range("full"))
    frame.SetTitle("")
    if args.blind is None:
        data.plotOn(frame, R.RooFit.Binning(args.bins), R.RooFit.Name("Data"))
    else:
        blind_lo, blind_hi = args.blind
        m.setRange("full",  args.xmin, args.xmax)
        m.setRange("sb_lo", args.xmin, blind_lo)
        m.setRange("sb_hi", blind_hi,  args.xmax)
        sb_union = "sb_lo,sb_hi"
        #m.setRange("unblind_lo", args.xmin, blind_lo)
        #m.setRange("unblind_hi", blind_hi, args.xmax)
        #data_lo = data.reduce(f"{m.GetName()}<{blind_lo}")
        #data_hi = data.reduce(f"{m.GetName()}>{blind_hi}")
        #data_lo.plotOn(frame, R.RooFit.Name("Data"))
        #data_hi.plotOn(frame)

        cut_sb = f"({m.GetName()}<{blind_lo}) || ({m.GetName()}>{blind_hi})"
        n_sb = data.sumEntries(cut_sb)  # uses weights if present
        data_sb = data.reduce(R.RooFit.Cut(cut_sb))
        data_sb.SetName("data_sb")
        print("Sideband entries:", data_sb.numEntries())
        data_sb.plotOn(frame, R.RooFit.Name("Data"), R.RooFit.DrawOption("PE0"))

    for name, pdf in models:
        col, sty = style[name]
        if args.blind is None:
            pdf.plotOn(frame,
                       R.RooFit.Range("fit"),
                       R.RooFit.LineColor(col),
                       R.RooFit.LineStyle(sty),
                       R.RooFit.LineWidth(3),
                       R.RooFit.Name(name))
        else:
            pdf.plotOn(frame,
                       R.RooFit.Range("full"),           # draw everywhere
                       R.RooFit.NormRange(sb_union),     # normalize only to sidebands
                       R.RooFit.Normalization(n_sb, R.RooAbsReal.NumEvent),  # match SB yield
                       R.RooFit.LineColor(col),
                       R.RooFit.LineStyle(sty),
                       R.RooFit.LineWidth(3),
                       R.RooFit.Name(name))

    # Legend
    leg = R.TLegend(0.60, 0.57, 0.90, 0.90)
    leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextFont(42)
    leg.AddEntry(frame.getObject(0), "Data", "lep")
    for i in range(1, int(frame.numItems())):
        obj = frame.getObject(i)
        if obj:
            leg.AddEntry(obj, obj.GetName(), "l")

    # Ratio graphs (PDF / reference)
    ref_name = args.ratio_ref
    ref_curve = find_curve_by_name(frame, ref_name)
    if ref_curve is None:
        names = [frame.getObject(i).GetName() for i in range(int(frame.numItems())) if frame.getObject(i)]
        raise RuntimeError(f"Ratio reference '{ref_name}' not found on frame. Available: {names}")

    ratio_graphs = []
    for name, _pdf in models:
        if args.no_ratio: break
        if name == ref_name:
            flat = R.TGraph()
            flat.SetName(f"ratio_{name}")
            flat.SetLineColor(style[name][0]); flat.SetLineStyle(style[name][1]); flat.SetLineWidth(3)
            xbuf = array('d',[0.0]); ybuf = array('d',[0.0])
            for i in range(ref_curve.GetN()):
                ref_curve.GetPoint(i, xbuf, ybuf)
                flat.SetPoint(flat.GetN(), float(xbuf[0]), 1.0)
            ratio_graphs.append((name, flat))
            continue
        num_curve = find_curve_by_name(frame, name)
        if not num_curve: continue
        col, sty = style[name]
        ratio_graphs.append((name, graph_ratio(num_curve, ref_curve, f"ratio_{name}", col, sty)))

    # Canvas
    cname = f"c_{label}"
    if args.no_ratio:
        c = R.TCanvas(cname, cname, 900, 700)
        c.SetLeftMargin(0.12); c.SetRightMargin(0.04)
        c.SetBottomMargin(0.12); c.SetTopMargin(0.06)
        frame.GetYaxis().SetTitle("Events / GeV")
        frame.Draw(); leg.Draw()
    else:
        c = R.TCanvas(cname, cname, 900, 900)
        pad1 = R.TPad("pad1","pad1",0,0.28,1,1)
        pad2 = R.TPad("pad2","pad2",0,0.00,1,0.28)
        for p in (pad1, pad2):
            p.SetLeftMargin(0.12); p.SetRightMargin(0.04)
        pad1.SetBottomMargin(0.02)
        pad2.SetTopMargin(0.05); pad2.SetBottomMargin(0.40); pad2.SetGridy(True)
        pad1.Draw(); pad2.Draw()

        pad1.cd()
        frame.GetXaxis().SetTitle("")
        frame.GetXaxis().SetLabelSize(0) # removes numbers (110,115,…150)
        frame.GetXaxis().SetTitleSize(0) # removes "m_{μμ}" or whatever
        frame.GetYaxis().SetTitle("Events / GeV")
        frame.GetYaxis().SetTitleOffset(1.2)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.Draw(); leg.Draw()

        pad2.cd()
        hframe = R.TH1F("hframe",";m_{#mu#mu} [GeV];Ratio PDFs", 100, args.xmin, args.xmax)
        hframe.SetStats(0)
        hframe.SetMinimum(0.98); hframe.SetMaximum(1.02)
        hframe.GetXaxis().SetTitleSize(0.14)
        hframe.GetXaxis().SetLabelSize(0.12)
        hframe.GetXaxis().SetTitleOffset(1.0)
        hframe.GetYaxis().SetTitleSize(0.12)
        hframe.GetYaxis().SetLabelSize(0.11)
        hframe.GetYaxis().SetTitleOffset(0.5)
        hframe.GetYaxis().SetNdivisions(505)
        hframe.Draw("AXIS")
        for _, gr in ratio_graphs: gr.Draw("L SAME")

    # Labels
    c.cd()
    latex = R.TLatex(); latex.SetNDC(True)
    latex.SetTextFont(61); latex.SetTextSize(0.05); latex.DrawLatex(0.14, 0.93, "CMS")
    latex.SetTextFont(42); latex.SetTextSize(0.04)
    latex.DrawLatex(0.62, 0.93, f"{args.lumi:.1f} fb^{{-1}} (13.6 TeV)")
    latex.SetTextSize(0.045); latex.DrawLatex(0.16, 0.85, f"{cat.GetName()} [{lo:.2g},{hi:.2g})")

    # Save plots
    outName = label.replace("[", "").replace(")", "").replace(",", "-")
    out_png = os.path.join(args.out, f"{outName}_v3_Blind.png")
    out_pdf = os.path.join(args.out, f"{outName}_v3_Blind.pdf")
    c.SaveAs(out_png); c.SaveAs(out_pdf)

    # Print chi2 table
    print(f"\n=== χ²/ndf (binning = {args.bins}, range = [{args.xmin:.1f},{args.xmax:.1f}]) for {label} ===")
    for name, chi2ndf, ndf in chi2_table:
        print(f"{name:18s}  χ²/ndf = {chi2ndf:.3f}  (ndf={ndf})")

    return data, models, results

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mass-branch", required=True)
    ap.add_argument("--weight-branch", default=None)
    ap.add_argument("--cat-var", required=True, help="Branch name for category variable (e.g. bdt_score_t)")
    ap.add_argument("--cat-edges", required=True, help="Comma-separated edges, e.g. 0,0.141,0.242,0.5443,0.6443,0.89,1.0")
    ap.add_argument("--xmin", type=float, default=110.0)
    ap.add_argument("--xmax", type=float, default=150.0)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--out", default="bkg_fit_v7_out")
    ap.add_argument("--ratio-ref", default="BWZRedux", help="Model name used as denominator for the ratio panel")
    ap.add_argument("--no-ratio", action="store_true", help="Disable the ratio panel")
    ap.add_argument("--lumi", type=float, default=62.3)  # fb^-1
    ap.add_argument("--blind", nargs=2, type=float, metavar=("LO","HI"),
                    help="If given, blind the mass window [LO,HI] in plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Parse category edges
    edges = [float(x) for x in args.cat_edges.split(",")]
    if len(edges) < 2:
        raise RuntimeError("Need at least two edges for categories")
    cat_min, cat_max = edges[0], edges[-1]

    # Open input
    f = R.TFile.Open(args.input)
    t = f.Get(args.tree)
    if not t: raise RuntimeError("Tree not found")

    # Observables
    m = R.RooRealVar(args.mass_branch, "m_{#mu#mu} [GeV]", args.xmin, args.xmax)
    m.setRange("fit", args.xmin, args.xmax)
    m.setBins(args.bins, "fit")

    if not has_branch(t, args.cat_var):
        raise RuntimeError(f"Branch '{args.cat_var}' not found in tree")
    cat = R.RooRealVar(args.cat_var, args.cat_var, cat_min, cat_max)

    # Build dataset including cat (and weight if present)
    cut_mass = f"({args.mass_branch}>={args.xmin} && {args.mass_branch}<={args.xmax})"
    vars_set = R.RooArgSet(m, cat)
    wname = args.weight_branch or autodetect_weight(t)
    if wname and has_branch(t, wname):
        w = R.RooRealVar(wname, wname, -1e9, 1e9)
        vars_set.add(w)
        data_full = R.RooDataSet("full", "full", t, vars_set, cut_mass, wname)
        print(f"Using weighted dataset with '{wname}'")
    else:
        data_full = R.RooDataSet("full", "full", t, vars_set, cut_mass)
        print("Using unweighted dataset.")

    print(f"Entries inclusive: {t.GetEntries()} | used in window: {data_full.numEntries()}")

    # Loop categories
    all_results = {}
    wsp = R.RooWorkspace("w","w")
    getattr(wsp,"import")(m); getattr(wsp,"import")(cat); getattr(wsp,"import")(data_full)

    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        data_cat, models, results = do_category(i, lo, hi, data_full, m, cat, args)

        # Import PDFs + fit results for this category (with unique names)
        for name, pdf in models:
            getattr(wsp, "import")(pdf, f"{name}_cat{i}")
        for name, res in results.items():
            if hasattr(wsp, "Import"):
                wsp.Import(res, f"res_{name}_cat{i}")
            else:
                getattr(wsp, "import")(res, f"res_{name}_cat{i}")

        all_results[i] = (data_cat, models, results)

    wsp.writeToFile(os.path.join(args.out, "fit_workspace.root"))
    print(f"\nSaved outputs under: {args.out}/")

if __name__ == "__main__":
    main()
