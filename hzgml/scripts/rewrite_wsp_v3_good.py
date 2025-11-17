#!/usr/bin/env python3
import sys, ROOT

def _wimp(ws, obj, *cmds):
    # call the C++ method named "import"
    return getattr(ws, "import")(obj, *cmds)

def _to_pylist_cpp_list(lst):
    """Make a Python list out of a C++ std::list<T*> with PyROOT 6.30."""
    try:
        return list(lst)  # works with recent pythonizations
    except TypeError:
        out = []
        it = lst.begin()
        while it != lst.end():
            out.append(it.__deref__())
            it.__preinc__()
        return out

def _iter_rooargset(rset):
    """Iterate a RooArgSet safely across ROOT versions."""
    it = rset.fwdIterator()
    while True:
        obj = it.next()
        if not obj: break
        yield obj

def rewrite_workspace(infile, wsname, outfile, wsout_name="w"):
    # Load Combine first, so RooMultiPdf / BernsteinFast, etc. have dictionaries
    try:
        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
    except Exception:
        pass

    fin = ROOT.TFile.Open(infile, "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open input file: {infile}")

    win = fin.Get(wsname)
    if not win:
        raise RuntimeError(f"Workspace '{wsname}' not found in {infile}")

    # Create a brand new workspace
    wout = ROOT.RooWorkspace(wsout_name, wsout_name)

    # Helper to import with conflict recycling
    recycle = ROOT.RooFit.RecycleConflictNodes(True)

    # 1) Datasets (this will also pull in the observables automatically)
    data_list = _to_pylist_cpp_list(win.allData())
    for d in data_list:
        if not d: continue
        _wimp(wout, d, recycle)

    # 2) PDFs (dependencies get pulled in automatically)
    pdf_list = _to_pylist_cpp_list(win.allPdfs())
    for p in pdf_list:
        if not p: continue
        # import directly; cloneTree is unnecessary and can drag caches
        _wimp(wout, p, recycle)

    # 3) Categories — e.g. MultiPdf index categories
    try:
        cats = win.allCats()
        for c in _iter_rooargset(cats):
            if not c: continue
            if not wout.obj(c.GetName()):
                _wimp(wout, c, recycle)
    except Exception:
        pass  # not all ROOT builds expose allCats()

    # 4) Variables that Combine may expect (norms, nuisance, etc.)
    try:
        vars_rs = win.allVars()
        for v in _iter_rooargset(vars_rs):
            if not v: continue
            # Only bring what isn't already there via (1)–(3)
            if not wout.obj(v.GetName()):
                _wimp(wout, v, recycle)
    except Exception:
        pass

    # 5) DO NOT import functions/generic objects → this is where NLLs live.
    #    Skipping win.allFunctions() and win.allGenericObjects() on purpose.

    # Optional: a tiny “prime” to load dictionaries (harmless)
    try:
        _ = wout.var("mh_ggh")
        _ = wout.pdf("bkg_cat0_ggh_pdf")
    except Exception:
        pass

    # Write only the clean workspace
    wout.writeToFile(outfile, True)
    print(f"[ok] wrote cleaned workspace to: {outfile}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: rewrite_wsp_clean.py <input.root> <wsname> <output.root> [output_wsname]")
        sys.exit(1)
    infile, wsname, outfile = sys.argv[1:4]
    wsout_name = sys.argv[4] if len(sys.argv) > 4 else "w"
    rewrite_workspace(infile, wsname, outfile, wsout_name)
