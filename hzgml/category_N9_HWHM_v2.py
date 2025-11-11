# -*- coding: utf-8 -*-
# Optimization binning with per-category HWHM (for optimization) and
# a smooth "fixed-window" significance curve for plotting.

import math
import uuid
import ROOT
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# ----- CONFIGURATION ----
# ------------------------
# trained with sig=ggH+VBF
#sig_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0717_bdt_0716_2223/ggH/sig.root"
#bkg_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0717_bdt_0716_2223/ggH/bkg.root"
# trained with sig =ggH
sig_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/sig.root"
bkg_file   = "/eos/user/q/qguo/vbfhmm/ml/RunIII/skimmed_ntuples_ggH_v1/bdt_0820_bdt_0820_2223/ggH/bkg.root"
tree_name  = "test"
score_var  = "bdt_score"
weight_var = "eventWeight"
mass_var   = "diMufsr_rc_mass"

num_thresholds = 200    # denser -> smoother曲线
max_categories = 5

# mass histogram settings used to find FWHM/HWHM
mass_nbins = 160
mass_min   = 110.0
mass_max   = 150.0

# FWHM护栏，避免离谱宽度（可按你的模型细化）
MIN_FWHM = 1.5  # GeV
MAX_FWHM = 8.0  # GeV

random_seed = 12345
np.random.seed(random_seed)

# ------------------------
# ----- UTILITIES --------
# ------------------------
def build_mass_hist(masses, weights, nbins=mass_nbins, xmin=mass_min, xmax=mass_max, name_prefix="hmass"):
    mask = np.isfinite(masses) & np.isfinite(weights)
    masses  = masses[mask]
    weights = weights[mask]
    hname = f"{name_prefix}_{uuid.uuid4().hex}"
    hist = ROOT.TH1F(hname, hname, nbins, xmin, xmax)
    hist.Sumw2()
    for m, w in zip(masses, weights):
        hist.Fill(float(m), float(w))
    hist.SetDirectory(0)
    return hist

def _gauss_fit_mu_fwhm(h, m0=None, halfwin=5.0):
    if h.GetEntries() < 10 or h.Integral() <= 0:
        return None, None
    if m0 is None:
        m0 = h.GetBinCenter(h.GetMaximumBin())
    lo = max(mass_min, m0 - halfwin)
    hi = min(mass_max, m0 + halfwin)
    if hi <= lo:
        return None, None
    f = ROOT.TF1(f"g_{uuid.uuid4().hex}", "gaus", lo, hi)
    f.SetParameters(h.GetMaximum(), m0, 1.5)
    status = h.Fit(f, "QN0")
    if status != 0:
        return None, None
    mu = f.GetParameter(1)
    sigma = abs(f.GetParameter(2))
    if not (0.2 <= sigma <= 6.0):
        return None, None
    fwhm = 2.354820045 * sigma
    return mu, fwhm

def _halfmax_mu_fwhm(h, smooth_times=3):
    if h.GetEntries() < 10 or h.Integral() <= 0:
        return None, None
    hs = h.Clone(f"{h.GetName()}_sm")
    hs.SetDirectory(0)
    for _ in range(max(0, smooth_times)):
        hs.Smooth()
    ib = hs.GetMaximumBin()
    ymax = hs.GetBinContent(ib)
    if ymax <= 0:
        return None, None
    target = 0.5 * ymax
    mu = hs.GetBinCenter(ib)

    # left crossing
    i = ib
    while i > 1 and hs.GetBinContent(i) >= target:
        i -= 1
    if i == 1 and hs.GetBinContent(i) >= target:
        m_left = mass_min
    else:
        x1, y1 = hs.GetBinCenter(i), hs.GetBinContent(i)
        x2, y2 = hs.GetBinCenter(i+1), hs.GetBinContent(i+1)
        m_left = x1 if y2 == y1 else x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    # right crossing
    i = ib
    nb = hs.GetNbinsX()
    while i < nb and hs.GetBinContent(i) >= target:
        i += 1
    if i == nb and hs.GetBinContent(i) >= target:
        m_right = mass_max
    else:
        x1, y1 = hs.GetBinCenter(i-1), hs.GetBinContent(i-1)
        x2, y2 = hs.GetBinCenter(i),   hs.GetBinContent(i)
        m_right = x2 if y2 == y1 else x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    fwhm = max(0.0, m_right - m_left)
    if fwhm <= 0:
        return None, None
    return mu, fwhm

def _clamp_fwhm(fwhm):
    return max(MIN_FWHM, min(MAX_FWHM, fwhm))

def robust_mu_hwhm_from_hist(h, mu_fallback=None, fwhm_fallback=None):
    mu, fwhm = _gauss_fit_mu_fwhm(h)
    if mu is None or fwhm is None:
        mu2, fwhm2 = _halfmax_mu_fwhm(h, smooth_times=3)
        mu   = mu if mu is not None else mu2
        fwhm = fwhm if fwhm is not None else fwhm2
    if (mu is None or fwhm is None) and (mu_fallback is not None and fwhm_fallback is not None):
        mu, fwhm = mu_fallback, fwhm_fallback
    if mu is None or fwhm is None:
        mu, fwhm = 125.0, 4.0
    fwhm = _clamp_fwhm(fwhm)
    hwhm = 0.5 * fwhm
    lo = max(mass_min, mu - hwhm)
    hi = min(mass_max, mu + hwhm)
    if hi <= lo:  # 极端回退
        lo, hi = max(mass_min, mu - 2.0), min(mass_max, mu + 2.0)
    return mu, hwhm, lo, hi, fwhm

def compute_global_window(sig_masses, sig_weights):
    h = build_mass_hist(sig_masses, sig_weights, name_prefix="h_sig_global")
    mu, hwhm, lo, hi, fwhm = robust_mu_hwhm_from_hist(h)
    return (lo, hi), mu, fwhm

# ------------------------
# ----- I/O --------------
# ------------------------
def load_data(file, tree=tree_name, score_var=score_var, weight_var=weight_var, mass_var=mass_var):
    f = ROOT.TFile.Open(file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {file}")
    t = f.Get(tree)
    if not t:
        raise RuntimeError(f"Cannot find tree '{tree}' in {file}")
    scores, weights, masses = [], [], []
    for ev in t:
        scores.append(getattr(ev, score_var))
        weights.append(getattr(ev, weight_var))
        masses.append(getattr(ev, mass_var))
    return np.asarray(scores, float), np.asarray(weights, float), np.asarray(masses, float)

# ------------------------
# ----- METRICS ----------
# ------------------------
def asimov_sig(S, B):
    if B <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0*((S+B)*math.log(1.0 + S/max(B, 1e-300)) - S)))

def compute_significance(sig_scores, sig_weights, sig_masses,
                         bkg_scores, bkg_weights, bkg_masses,
                         boundaries,
                         mode="per_category",
                         global_window=None,
                         global_mu=None, global_fwhm=None):
    """
    mode:
      - "per_category": 每个类别用自身信号得到的 HWHM 作为窗口（用于优化）
      - "global": 所有类别用同一个固定窗口 global_window=(lo,hi)（用于平滑画图）
    """
    S_list, B_list = [], []
    lower = -np.inf
    for boundary in list(boundaries) + [np.inf]:
        mask_sig_cat = (sig_scores > lower) & (sig_scores <= boundary)
        mask_bkg_cat = (bkg_scores > lower) & (bkg_scores <= boundary)

        if mode == "global":
            lo, hi = global_window
        else:  # per_category
            h = build_mass_hist(sig_masses[mask_sig_cat], sig_weights[mask_sig_cat],
                                name_prefix="hsig_cat")
            mu, hwhm, lo, hi, _ = robust_mu_hwhm_from_hist(h, mu_fallback=global_mu, fwhm_fallback=global_fwhm)

        mask_sig = mask_sig_cat & (sig_masses >= lo) & (sig_masses <= hi)
        mask_bkg = mask_bkg_cat & (bkg_masses >= lo) & (bkg_masses <= hi)

        S = float(sig_weights[mask_sig].sum())
        B = float(bkg_weights[mask_bkg].sum())
        S_list.append(S); B_list.append(B)
        lower = boundary

    return math.sqrt(sum(asimov_sig(S, B)**2 for S, B in zip(S_list, B_list)))

# ------------------------
# ----- SCAN -------------
# ------------------------
def iterative_binning(sig_scores, sig_weights, sig_masses,
                      bkg_scores, bkg_weights, bkg_masses,
                      num_thresholds=100, max_categories=5):
    """
    迭代建类（用于决定最佳边界），优化目标用 per_category 窗口的显著度。
    同时，为画图准备：在相同阈值上再计算一份 global 窗口的显著度（平滑）。
    """
    # 准备全局窗口（平滑曲线用）
    (g_lo, g_hi), g_mu, g_fwhm = compute_global_window(sig_masses, sig_weights)

    category_bounds = []
    thresholds = np.linspace(min(sig_scores.min(), bkg_scores.min()),
                             max(sig_scores.max(), bkg_scores.max()),
                             num_thresholds+2)[1:-1]
    scan_history = []

    # 起始：1个类别
    Z0_dyn = compute_significance(sig_scores, sig_weights, sig_masses,
                                  bkg_scores, bkg_weights, bkg_masses,
                                  category_bounds, mode="per_category",
                                  global_window=(g_lo, g_hi), global_mu=g_mu, global_fwhm=g_fwhm)
    Z0_fix = compute_significance(sig_scores, sig_weights, sig_masses,
                                  bkg_scores, bkg_weights, bkg_masses,
                                  category_bounds, mode="global",
                                  global_window=(g_lo, g_hi))
    scan_history.append({
        "x_eff":[], "x_bdt":[],
        "Z_dyn":[], "Z_fix":[],
        "prev_eff":[], "prev_bdt":[],
        "bestZ_dyn":Z0_dyn, "eff_opt":None, "bdt_opt":None
    })

    for n_cat in range(2, max_categories+1):
        best_Z_dyn = -np.inf
        best_new_boundary = None
        best_split_idx = None

        x_eff = []; x_bdt = []
        Z_dyn = []; Z_fix = []
        split_idx = []

        all_bins = [-np.inf] + category_bounds + [np.inf]
        for i_bin in range(len(all_bins)-1):
            low, up = all_bins[i_bin], all_bins[i_bin+1]
            t_in = thresholds[(thresholds > low) & (thresholds < up)]
            for thr in t_in:
                new_bounds = sorted(category_bounds + [float(thr)])

                # per-category 窗口（用于优化）
                Zd = compute_significance(sig_scores, sig_weights, sig_masses,
                                          bkg_scores, bkg_weights, bkg_masses,
                                          new_bounds, mode="per_category",
                                          global_window=(g_lo, g_hi), global_mu=g_mu, global_fwhm=g_fwhm)
                # global 固定窗口（用于平滑画图）
                Zg = compute_significance(sig_scores, sig_weights, sig_masses,
                                          bkg_scores, bkg_weights, bkg_masses,
                                          new_bounds, mode="global",
                                          global_window=(g_lo, g_hi))

                eff = sig_weights[sig_scores > thr].sum() / max(sig_weights.sum(), 1e-300)
                x_eff.append(1.0 - eff); x_bdt.append(float(thr))
                Z_dyn.append(float(Zd)); Z_fix.append(float(Zg)); split_idx.append(i_bin)

                if Zd > best_Z_dyn:
                    best_Z_dyn = Zd
                    best_new_boundary = float(thr)
                    best_split_idx = i_bin

        eff_opt = 1.0 - sig_weights[sig_scores > best_new_boundary].sum() / max(sig_weights.sum(), 1e-300)
        category_bounds.insert(best_split_idx, best_new_boundary)
        category_bounds = sorted(set(category_bounds))

        prev_eff_bounds = [1.0 - sig_weights[sig_scores > b].sum() / max(sig_weights.sum(), 1e-300)
                           for b in category_bounds if b != best_new_boundary]
        prev_bdt_bounds = [b for b in category_bounds if b != best_new_boundary]

        scan_history.append({
            "x_eff":x_eff, "x_bdt":x_bdt,
            "Z_dyn":Z_dyn, "Z_fix":Z_fix,
            "prev_eff":prev_eff_bounds, "prev_bdt":prev_bdt_bounds,
            "bestZ_dyn":best_Z_dyn, "eff_opt":eff_opt, "bdt_opt":best_new_boundary
        })

    # 把全局窗口也返回，便于打印/记录
    return scan_history, (g_lo, g_hi), g_mu, g_fwhm

# ------------------------
# ----- PLOTS ------------
# ------------------------
def plot_iterative_scan(scan_history, max_categories=5):
    for n_cat in range(2, max_categories+1):
        entry = scan_history[n_cat-1]
        x_eff = np.array(entry["x_eff"])
        Z_fix = np.array(entry["Z_fix"])
        Z_dyn = np.array(entry["Z_dyn"])
        prev_eff = entry["prev_eff"]
        eff_opt  = entry["eff_opt"]

        # 排序后绘制（确保横轴单调 -> 连续曲线）
        idx = np.argsort(x_eff)
        xe = x_eff[idx]; Zf = Z_fix[idx]; Zd = Z_dyn[idx]

        # 平滑曲线（固定窗口）——主图
        plt.figure(figsize=(6,6))
        plt.plot(xe, Zf, 'k.', markersize=3, label='Scan (fixed window)')
        for b in prev_eff:
            plt.axvline(b, color='gray', linestyle='-', linewidth=1)
        if eff_opt is not None:
            plt.axvline(eff_opt, color='red', linestyle='--', linewidth=1.5, label='Optimal boundary')
        plt.xlabel('1 - $\\epsilon_{\\mathrm{sig}}$', fontsize=16)
        plt.ylabel('Significance ($\\sigma$)', fontsize=16)
        plt.title(f'Significance scan: {n_cat-1} → {n_cat} categories (fixed HWHM)', fontsize=15)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'check_significance_scan_eff_N_{n_cat}_fixed_bdt.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 可选：对照图（动态窗口，可能有轻微“抖动/不连续”）
        plt.figure(figsize=(6,6))
        plt.plot(xe, Zd, 'k.', markersize=3, label='Scan (per-category HWHM)')
        for b in prev_eff:
            plt.axvline(b, color='gray', linestyle='-', linewidth=1)
        if eff_opt is not None:
            plt.axvline(eff_opt, color='red', linestyle='--', linewidth=1.5)
        plt.xlabel('1 - $\\epsilon_{\\mathrm{sig}}$', fontsize=16)
        plt.ylabel('Significance ($\\sigma$)', fontsize=16)
        plt.title(f'Significance scan: {n_cat-1} → {n_cat} categories (dynamic HWHM)', fontsize=15)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'check_significance_scan_eff_N_{n_cat}_dynamic_bdt.png', dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------
# ----- MAIN -------------
# ------------------------
if __name__ == "__main__":
    # Load full samples (no mass cut)
    sig_scores, sig_weights, sig_masses = load_data(sig_file)
    bkg_scores, bkg_weights, bkg_masses = load_data(bkg_file)

    # Build categories iteratively (optimize with per-category HWHM),
    # while also preparing a fixed-window curve for plotting.
    scan_history, global_window, g_mu, g_fwhm = iterative_binning(
        sig_scores, sig_weights, sig_masses,
        bkg_scores, bkg_weights, bkg_masses,
        num_thresholds=num_thresholds,
        max_categories=max_categories
    )

    # Make diagnostic plots (smooth fixed-window + dynamic for reference)
    plot_iterative_scan(scan_history, max_categories=max_categories)

    # Report results
    print('\nFinal category boundaries (iterative history):')
    for i, entry in enumerate(scan_history):
        if i < 1:
            print(f"After {i+1} category:")
        else:
            print(f"After {i+1} categories:")
        bdt_opt = entry["bdt_opt"]
        eff_opt = entry["eff_opt"]
        print(f"  BDT optimal cut     = {None if bdt_opt is None else round(bdt_opt, 6)}")
        print(f"  1 - epsilon_sig     = {None if eff_opt is None else round(eff_opt, 6)}")
        print(f"  Best Z (dynamic)    = {entry['bestZ_dyn']:.3f}")
    print(f"\nGlobal fixed window used for smooth curve: [{global_window[0]:.3f}, {global_window[1]:.3f}] GeV "
          f"(mu≈{g_mu:.3f}, FWHM≈{g_fwhm:.3f})")
