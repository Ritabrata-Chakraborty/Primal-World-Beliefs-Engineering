"""Re-analysis of the Primal World Beliefs survey of Indian engineering students.

Input : data/responses.csv  (Google Form export, 108 timestamped responses)
Output: out/results.json, out/*.csv  (all numbers used in the manuscript)
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20241013)
R = {}

# ------------------------------------------------------------------ load
# Usage: python analysis.py                      -> original Google Form export (data/responses.csv)
#        python analysis.py <deidentified.csv> <outdir> -> shared de-identified file (state -> region, no time stamps)
PI = ["P_dull", "P_boring", "P_worth", "P_fascinating", "P_abundant", "P_beautyaround",
      "P_reason", "P_meant", "P_needsme", "P_helpme", "P_lackpurpose",
      "P_safe", "P_dangerous", "P_harmless", "P_good", "P_cutthroat", "P_beautyugly", "P_worse"]
ENG = ["ES1", "ES2", "ES3", "ES4", "ES5", "ES6",
       "EE1", "EE2", "EE3", "EE4", "EE5", "EE6",
       "EA1", "EA2", "EA3", "EA4", "EA5", "EA6"]
if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
    df["timestamp"] = pd.NaT; df["state"] = df["region"]
    if len(sys.argv) > 2: OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
else:
    raw = pd.read_csv(ROOT / "data/responses.csv", encoding="utf-8")
    cols = list(raw.columns)
    pi_cols, eng_cols = cols[4:22], cols[22:40]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw.iloc[:, 0], format="%d/%m/%Y %H:%M:%S"),
        "sex": raw.iloc[:, 1].astype(int),
        "state": raw.iloc[:, 2].astype(str).str.strip(),
        "age": raw["Age"].astype(int),
    })
    for c, n in zip(pi_cols, PI): df[n] = raw[c].astype(int)
    for c, n in zip(eng_cols, ENG): df[n] = raw[c].astype(int)
    df.to_csv(OUT / "clean_raw_coded.csv", index=False)

valid = {-5, -3, -1, 1, 3, 5}
assert all(df[c].isin(valid).all() for c in PI + ENG), "out-of-range item value"
assert df[PI + ENG].notna().all().all()

# Validated PI-18 key (Clifton & Yaden, 2021; administration instructions, Feb 2021)
REV = ["P_dull", "P_boring", "P_worth", "P_lackpurpose", "P_dangerous", "P_cutthroat", "P_worse"]
SAFE = ["P_safe", "P_dangerous", "P_cutthroat", "P_worse", "P_good", "P_harmless"]
ENTICING = ["P_beautyugly", "P_worth", "P_boring", "P_abundant", "P_fascinating", "P_dull", "P_beautyaround"]
ALIVE = ["P_helpme", "P_meant", "P_lackpurpose", "P_needsme", "P_reason"]
GOOD = SAFE + ENTICING + ["P_helpme", "P_needsme"]
E_SAFE, E_ENT, E_ALIVE = ENG[:6], ENG[6:12], ENG[12:]
assert len(set(SAFE + ENTICING + ALIVE)) == 18

# 0-5 metric (validated scoring): raw -5,-3,-1,1,3,5 -> 0..5 ; reversed = 5 - x
s = pd.DataFrame(index=df.index)
for c in PI + ENG: s[c] = (df[c] + 5) / 2
for c in REV: s[c] = 5 - s[c]
SC = {"Safe": SAFE, "Enticing": ENTICING, "Alive": ALIVE, "Good": GOOD,
      "Eng_Safe": E_SAFE, "Eng_Enticing": E_ENT, "Eng_Alive": E_ALIVE, "Eng_Total": ENG}
for k, v in SC.items(): s[k] = s[v].mean(axis=1)

# ------------------------------------------------------------------ sample
R["N"] = int(len(df))
R["sex_counts"] = {str(k): int(v) for k, v in df.sex.value_counts().items()}
R["age"] = dict(M=df.age.mean(), SD=df.age.std(), min=int(df.age.min()), max=int(df.age.max()),
                median=float(df.age.median()),
                share_17_20=float((df.age <= 20).mean()))
R["age_counts"] = {int(k): int(v) for k, v in df.age.value_counts().sort_index().items()}
st = df.state.value_counts()
R["n_states"] = int(st.size); R["state_counts"] = {k: int(v) for k, v in st.items()}
R["window"] = [str(df.timestamp.min()), str(df.timestamp.max())]

# ------------------------------------------------------------------ helpers
def alpha(X):
    X = np.asarray(X, float); k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))

def omega_total(X):
    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml"); fa.fit(X)
    l = fa.loadings_[:, 0]; l = l if l.sum() > 0 else -l
    u = 1 - l ** 2
    return (l.sum() ** 2) / (l.sum() ** 2 + u.sum()), l

def boot_ci(fun, X, B=4000):
    X = np.asarray(X); n = len(X); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try: vals.append(fun(X[idx]))
        except Exception: pass
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

def citc(X):
    X = pd.DataFrame(X); return {c: X[c].corr(X.drop(columns=c).sum(axis=1)) for c in X}

def r_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

def mean_ci(x):
    x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(.975, len(x) - 1); return [float(m - t * se), float(m + t * se)]

# ------------------------------------------------------------------ data quality
long = []
for i, row in df[PI + ENG].iterrows():
    v = row.values; best = cur = 1
    for a, b in zip(v[:-1], v[1:]):
        cur = cur + 1 if a == b else 1; best = max(best, cur)
    long.append(best)
df["longstring"] = long
df["irv"] = df[PI + ENG].std(axis=1)
# acquiescence: mean of raw (unreversed) responses, balanced across keying direction
FWD = [c for c in PI if c not in REV]
df["ars_fwd"] = df[FWD].mean(axis=1); df["ars_rev"] = df[REV].mean(axis=1)
df["ARS"] = (df.ars_fwd + df.ars_rev) / 2           # -5..5 metric, 0 = balanced
s["ARS"] = df["ARS"] / 2                            # 0-5-unit metric (scale points / 2)
# antonym pairs answered on the same side (both agree or both disagree)
PAIRS = [("P_safe", "P_dangerous"), ("P_harmless", "P_dangerous"), ("P_fascinating", "P_dull"),
         ("P_fascinating", "P_boring"), ("P_abundant", "P_worth"), ("P_reason", "P_lackpurpose"),
         ("P_meant", "P_lackpurpose"), ("P_good", "P_worse")]
same = np.zeros(len(df))
for a, b in PAIRS: same += (np.sign(df[a]) == np.sign(df[b])).astype(int)
df["same_side_pairs"] = same
LS_CUT = 18   # identical answers to >= half of the 36 items in a row
df["flag_longstring"] = df.longstring >= LS_CUT
R["quality"] = dict(
    longstring_cut=LS_CUT, n_flag_longstring=int(df.flag_longstring.sum()),
    longstring_desc=df.longstring.describe().to_dict(),
    flagged_rows=df.loc[df.flag_longstring, ["timestamp", "longstring", "irv"]].astype(str).to_dict("records"),
    ARS_M=float(df.ARS.mean()), ARS_SD=float(df.ARS.std()), ARS_CI=mean_ci(df.ARS),
    ARS_t=float(stats.ttest_1samp(df.ARS, 0).statistic), ARS_p=float(stats.ttest_1samp(df.ARS, 0).pvalue),
    ARS_d=float(df.ARS.mean() / df.ARS.std()),
    ARS_fwd_M=float(df.ars_fwd.mean()), ARS_rev_M=float(df.ars_rev.mean()),
    share_ARS_pos=float((df.ARS > 0).mean()),
    same_side_M=float(df.same_side_pairs.mean()), same_side_SD=float(df.same_side_pairs.std()),
    same_side_dist={int(k): int(v) for k, v in df.same_side_pairs.value_counts().sort_index().items()},
    share_same_side_ge4=float((df.same_side_pairs >= 4).mean()),
)
# per-pair same-side rate
R["quality"]["pair_same_side_rate"] = {f"{a}|{b}": float((np.sign(df[a]) == np.sign(df[b])).mean()) for a, b in PAIRS}

# ------------------------------------------------------------------ item descriptives
items_tab = []
for c in PI + ENG:
    x = s[c]
    items_tab.append(dict(item=c, reverse=c in REV, M=x.mean(), SD=x.std(), skew=stats.skew(x),
                          kurt=stats.kurtosis(x), pct_agree=float(((df[c] > 0)).mean()),
                          raw_M=df[c].mean()))
items_tab = pd.DataFrame(items_tab); items_tab.to_csv(OUT / "item_descriptives.csv", index=False)

# ------------------------------------------------------------------ reliability
rel = {}
for k, v in SC.items():
    X = s[v].values
    om, l = omega_total(X)
    rel[k] = dict(k=len(v), alpha=alpha(X), alpha_CI=boot_ci(alpha, X, 3000), omega=om,
                  M=s[k].mean(), SD=s[k].std(), M_CI=mean_ci(s[k]),
                  mean_interitem_r=float(np.mean(np.corrcoef(X.T)[np.triu_indices(len(v), 1)])),
                  citc_min=float(min(citc(s[v]).values())), citc_max=float(max(citc(s[v]).values())),
                  skew=float(stats.skew(s[k])), share_above_mid=float((s[k] > 2.5).mean()))
R["reliability"] = rel
R["citc"] = {k: {c: float(r) for c, r in citc(s[v]).items()} for k, v in SC.items()}
# forward vs reversed composites inside each PI-18 subscale
fr = {}
for k, v in [("Safe", SAFE), ("Enticing", ENTICING), ("Alive", ALIVE)]:
    f = [c for c in v if c not in REV]; r_ = [c for c in v if c in REV]
    rr = s[f].mean(axis=1).corr(s[r_].mean(axis=1))
    fr[k] = dict(n_fwd=len(f), n_rev=len(r_), r=float(rr), CI=r_ci(rr, len(df)),
                 alpha_fwd=alpha(s[f]) if len(f) > 1 else None,
                 alpha_rev=alpha(s[r_]) if len(r_) > 1 else None)
allf = s[[c for c in PI if c not in REV]].mean(axis=1); allr = s[REV].mean(axis=1)
fr["all"] = dict(r=float(allf.corr(allr)), CI=r_ci(allf.corr(allr), len(df)))
R["fwd_rev"] = fr

# ------------------------------------------------------------------ EFA / parallel analysis
def parallel(X, B=1000, method="pca"):
    X = np.asarray(X, float); n, p = X.shape
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    sims = np.array([np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)).T))[::-1] for _ in range(B)])
    thr = np.percentile(sims, 95, axis=0)
    k = 0
    for e, t in zip(ev, thr):
        if e > t: k += 1
        else: break
    return ev, thr, k

efa = {}
for name, items in [("PI18", PI), ("ENG18", ENG)]:
    X = s[items]
    ev, thr, k = parallel(X)
    kmo_all, kmo_model = calculate_kmo(X); chi, p = calculate_bartlett_sphericity(X)
    efa[name] = dict(eigen=ev.tolist(), pa_95=thr.tolist(), n_factors_PA=k,
                     n_kaiser=int((ev > 1).sum()), var1=float(ev[0] / len(items)),
                     ratio_1_2=float(ev[0] / ev[1]), KMO=float(kmo_model), bartlett_chi2=float(chi),
                     bartlett_p=float(p))
    for nf in sorted({1, 3, max(k, 1)}):
        fa = FactorAnalyzer(n_factors=nf, rotation=("oblimin" if nf > 1 else None), method="minres")
        fa.fit(X)
        L = pd.DataFrame(fa.loadings_, index=items, columns=[f"F{j+1}" for j in range(nf)])
        L.to_csv(OUT / f"efa_{name}_{nf}f.csv")
        efa[name][f"loadings_{nf}f"] = L.round(3).to_dict()
        if nf > 1 and getattr(fa, "phi_", None) is not None:
            efa[name][f"phi_{nf}f"] = np.round(fa.phi_, 3).tolist()
        efa[name][f"var_{nf}f"] = [float(v) for v in fa.get_factor_variance()[1]]
R["efa"] = efa

# Schmid-Leiman style general-factor strength for ENG: omega hierarchical via bifactor-ish
def omega_h_SL(X, nf=3):
    fa = FactorAnalyzer(n_factors=nf, rotation="oblimin", method="minres"); fa.fit(X)
    L1, phi = fa.loadings_, fa.phi_
    fa2 = FactorAnalyzer(n_factors=1, rotation=None, method="minres", is_corr_matrix=True); fa2.fit(phi)
    g2 = fa2.loadings_[:, 0]; g2 = g2 if g2.sum() > 0 else -g2
    g = L1 @ g2; spec = L1 * np.sqrt(np.clip(1 - g2 ** 2, 0, None))
    u = 1 - (g ** 2 + (spec ** 2).sum(axis=1))
    tot = g.sum() ** 2 + (spec.sum(axis=0) ** 2).sum() + u.sum()
    return dict(omega_h=float(g.sum() ** 2 / tot), omega_t=float((tot - u.sum()) / tot),
                ECV=float((g ** 2).sum() / ((g ** 2).sum() + (spec ** 2).sum())),
                g_loadings=dict(zip(X.columns, np.round(g, 3).tolist())))
R["SL_ENG"] = omega_h_SL(s[ENG])
R["SL_PI"] = omega_h_SL(s[PI])

# ------------------------------------------------------------------ CFA (semopy, ML)
import semopy
def cfa(desc, data, name):
    m = semopy.Model(desc); m.fit(data, obj="MLW")
    st_ = semopy.calc_stats(m).T.iloc[:, 0].to_dict()
    est = m.inspect(std_est=True)
    return dict(name=name, chi2=st_["chi2"], df=st_["DoF"], p=st_["chi2 p-value"], CFI=st_["CFI"],
                TLI=st_["TLI"], RMSEA=st_["RMSEA"], AIC=st_["AIC"], BIC=st_["BIC"]), est

cfa_res = {}; cfa_est = {}
d = s.copy()
m_pi3 = f"Safe =~ {' + '.join(SAFE)}\nEnticing =~ {' + '.join(ENTICING)}\nAlive =~ {' + '.join(ALIVE)}"
m_pi3m = m_pi3 + f"\nMethod =~ {' + '.join(REV)}\nMethod ~~ 0*Safe\nMethod ~~ 0*Enticing\nMethod ~~ 0*Alive"
m_pi1 = f"G =~ {' + '.join(PI)}"
m_e3 = f"ESafe =~ {' + '.join(E_SAFE)}\nEEnt =~ {' + '.join(E_ENT)}\nEAlive =~ {' + '.join(E_ALIVE)}"
m_e1 = f"EG =~ {' + '.join(ENG)}"
for key, desc in [("PI_3f", m_pi3), ("PI_3f_method", m_pi3m), ("PI_1f", m_pi1),
                  ("ENG_3f", m_e3), ("ENG_1f", m_e1)]:
    try:
        r_, est = cfa(desc, d, key); cfa_res[key] = r_; cfa_est[key] = est
        est.to_csv(OUT / f"cfa_{key}.csv", index=False)
    except Exception as e:
        cfa_res[key] = dict(error=str(e))
# SRMR computed manually
def srmr(model_desc, data, items):
    m = semopy.Model(model_desc); m.fit(data[items + []] if False else data, obj="MLW")
    S = np.corrcoef(data[items].values.T)
    sig = m.calc_sigma()[0]
    obs = m.vars["observed"]; idx = [obs.index(c) for c in items]
    sig = sig[np.ix_(idx, idx)]; dsd = np.sqrt(np.diag(sig)); sigc = sig / np.outer(dsd, dsd)
    iu = np.tril_indices(len(items))
    return float(np.sqrt(np.mean((S[iu] - sigc[iu]) ** 2)))
for key, desc, items in [("PI_3f", m_pi3, PI), ("PI_1f", m_pi1, PI), ("ENG_3f", m_e3, ENG), ("ENG_1f", m_e1, ENG),
                         ("PI_3f_method", m_pi3m, PI)]:
    try: cfa_res[key]["SRMR"] = srmr(desc, d, items)
    except Exception as e: cfa_res[key]["SRMR_err"] = str(e)
# latent correlations
def latent_corr(est):
    e = est[(est.op == "~~") & (est.lval != est.rval)]
    return {f"{a}~{b}": float(v) for a, b, v in zip(e.lval, e.rval, e["Est. Std"])}
for key in ["PI_3f", "PI_3f_method", "ENG_3f"]:
    if key in cfa_est: cfa_res[key]["latent_corr"] = latent_corr(cfa_est[key])
# std loadings
for key in ["PI_3f", "PI_3f_method", "ENG_3f", "ENG_1f"]:
    if key in cfa_est:
        e = cfa_est[key]; e = e[e.op == "~"]
        cfa_res[key]["std_loadings"] = {f"{b}->{a}": round(float(v), 3) for a, b, v in zip(e.lval, e.rval, e["Est. Std"])}
# chi-square difference 1f vs 3f
for a, b in [("ENG_1f", "ENG_3f"), ("PI_1f", "PI_3f"), ("PI_3f", "PI_3f_method")]:
    try:
        dchi = cfa_res[a]["chi2"] - cfa_res[b]["chi2"]; ddf = cfa_res[a]["df"] - cfa_res[b]["df"]
        cfa_res[f"diff_{a}_vs_{b}"] = dict(dchi2=dchi, ddf=ddf, p=float(stats.chi2.sf(dchi, ddf)))
    except Exception: pass
R["cfa"] = cfa_res

# HTMT for ENG and PI subscales
def htmt(data, groups):
    C = data.corr().abs(); out = {}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = groups[names[i]], groups[names[j]]
            het = C.loc[a, b].values.mean()
            ma = C.loc[a, a].values[np.triu_indices(len(a), 1)].mean()
            mb = C.loc[b, b].values[np.triu_indices(len(b), 1)].mean()
            out[f"{names[i]}~{names[j]}"] = float(het / np.sqrt(ma * mb))
    return out
R["HTMT_ENG"] = htmt(s, {"Eng_Safe": E_SAFE, "Eng_Enticing": E_ENT, "Eng_Alive": E_ALIVE})
R["HTMT_PI"] = htmt(s, {"Safe": SAFE, "Enticing": ENTICING, "Alive": ALIVE})

# ------------------------------------------------------------------ correlations among scale scores
SCALES = ["Safe", "Enticing", "Alive", "Good", "Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total", "ARS", "age"]
s["age"] = df.age; s["sex"] = df.sex
C = s[SCALES].corr(); C.to_csv(OUT / "scale_correlations.csv")
P = pd.DataFrame(index=SCALES, columns=SCALES, dtype=float)
for a in SCALES:
    for b in SCALES: P.loc[a, b] = stats.pearsonr(s[a], s[b])[1] if a != b else np.nan
P.to_csv(OUT / "scale_correlations_p.csv")
R["corr"] = C.round(4).to_dict(); R["corr_p"] = P.round(5).to_dict()
# bootstrap CIs for key cross-domain correlations, and disattenuated
def bcorr(a, b, B=5000):
    x = s[[a, b]].values; n = len(x); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(np.corrcoef(x[i, 0], x[i, 1])[0, 1])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
cross = {}
for g in ["Safe", "Enticing", "Alive", "Good", "ARS"]:
    for e in ["Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
        r = s[g].corr(s[e]); ci = bcorr(g, e)
        dis = None
        if g in rel and e in rel: dis = r / np.sqrt(rel[g]["alpha"] * rel[e]["alpha"])
        cross[f"{g}~{e}"] = dict(r=float(r), CI=ci, p=float(stats.pearsonr(s[g], s[e])[1]),
                                 disattenuated=None if dis is None else float(dis))
R["cross"] = cross

# Is the engineering-domain score explained by general beliefs + acquiescence? (OLS with bootstrap)
def ols(y, Xcols, data):
    X = np.column_stack([np.ones(len(data))] + [data[c].values for c in Xcols]); yv = data[y].values
    b, *_ = np.linalg.lstsq(X, yv, rcond=None); res = yv - X @ b
    r2 = 1 - res.var() / yv.var()
    n, k = X.shape; s2 = res @ res / (n - k); cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    # standardized betas
    zb = b[1:] * np.array([data[c].std() for c in Xcols]) / data[y].std()
    t = b / se; p = 2 * stats.t.sf(np.abs(t), n - k)
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return dict(b=b.tolist(), se=se.tolist(), beta=zb.tolist(), p=p.tolist(), R2=float(r2), adjR2=float(adj),
                F=float((r2 / (k - 1)) / ((1 - r2) / (n - k))), df=[k - 1, n - k],
                Fp=float(stats.f.sf((r2 / (k - 1)) / ((1 - r2) / (n - k)), k - 1, n - k)))
reg = {}
for y in ["Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
    reg[y] = {"general_only": ols(y, ["Safe", "Enticing", "Alive"], s),
              "general_plus_ARS": ols(y, ["Safe", "Enticing", "Alive", "ARS"], s),
              "ARS_only": ols(y, ["ARS"], s)}
    # bootstrap CI for R2 of general_only
    v = []
    for _ in range(2000):
        i = rng.integers(0, len(s), len(s)); v.append(ols(y, ["Safe", "Enticing", "Alive"], s.iloc[i].reset_index(drop=True))["R2"])
    reg[y]["general_only"]["R2_CI"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R["regression"] = reg

# ------------------------------------------------------------------ domain vs general mean comparison
comp = {}
for g, e in [("Safe", "Eng_Safe"), ("Enticing", "Eng_Enticing"), ("Alive", "Eng_Alive"), ("Good", "Eng_Total")]:
    dlt = s[e] - s[g]
    t = stats.ttest_rel(s[e], s[g]); w = stats.wilcoxon(s[e], s[g])
    dz = dlt.mean() / dlt.std()
    # bootstrap CI for dz
    v = []
    x = dlt.values
    for _ in range(5000):
        i = rng.integers(0, len(x), len(x)); v.append(x[i].mean() / x[i].std(ddof=1))
    comp[f"{e}-{g}"] = dict(M_gen=s[g].mean(), SD_gen=s[g].std(), M_eng=s[e].mean(), SD_eng=s[e].std(),
                            diff=dlt.mean(), diff_CI=mean_ci(dlt), t=float(t.statistic), df=len(x) - 1,
                            p=float(t.pvalue), dz=float(dz), dz_CI=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                            wilcoxon_p=float(w.pvalue), share_eng_higher=float((dlt > 0).mean()),
                            share_gen_below_mid=float((s[g] < 2.5).mean()), share_eng_above_mid=float((s[e] > 2.5).mean()),
                            share_gen_below_eng_above=float(((s[g] < 2.5) & (s[e] > 2.5)).mean()),
                            n_gen_below_eng_above=int(((s[g] < 2.5) & (s[e] > 2.5)).sum()))
R["domain_vs_general"] = comp

# ------------------------------------------------------------------ comparison with published norms
# Clifton & Yaden (2021), Online Supplement Table S2, Study 2 (N = 5,171); items scored 0-5, reversed items reversed
NORM_ITEM = {"P_safe": 2.89, "P_dangerous": 2.72, "P_good": 3.43, "P_harmless": 2.65, "P_worse": 3.13,
             "P_cutthroat": 2.48, "P_dull": 3.87, "P_fascinating": 3.95, "P_boring": 3.68, "P_beautyugly": 3.84,
             "P_beautyaround": 3.96, "P_worth": 3.38, "P_abundant": 4.13, "P_reason": 3.03, "P_needsme": 2.96,
             "P_meant": 2.92, "P_lackpurpose": 2.95, "P_helpme": 3.17}
NORM_SD = {"P_safe": 1.30, "P_dangerous": 1.30, "P_good": 1.09, "P_harmless": 1.21, "P_worse": 1.22,
           "P_cutthroat": 1.28, "P_dull": 1.10, "P_fascinating": 1.00, "P_boring": 1.18, "P_beautyugly": 1.09,
           "P_beautyaround": 1.03, "P_worth": 1.31, "P_abundant": .95, "P_reason": 1.51, "P_needsme": 1.51,
           "P_meant": 1.45, "P_lackpurpose": 1.44, "P_helpme": 1.33}
norm = {}
for k, v in [("Safe", SAFE), ("Enticing", ENTICING), ("Alive", ALIVE), ("Good", GOOD)]:
    nm = float(np.mean([NORM_ITEM[c] for c in v]))
    t = stats.ttest_1samp(s[k], nm)
    norm[k] = dict(norm_M=nm, M=s[k].mean(), SD=s[k].std(), diff=s[k].mean() - nm, CI=mean_ci(s[k] - nm),
                   t=float(t.statistic), p=float(t.pvalue), d=float((s[k].mean() - nm) / s[k].std()))
norm_items = []
for c in PI:
    x = s[c]; t = stats.ttest_1samp(x, NORM_ITEM[c])
    pooled = np.sqrt((x.var() + NORM_SD[c] ** 2) / 2)
    norm_items.append(dict(item=c, reverse=c in REV, M=x.mean(), SD=x.std(), norm_M=NORM_ITEM[c], norm_SD=NORM_SD[c],
                           diff=x.mean() - NORM_ITEM[c], d=(x.mean() - NORM_ITEM[c]) / pooled, p=t.pvalue))
norm_items = pd.DataFrame(norm_items); norm_items.to_csv(OUT / "norm_item_comparison.csv", index=False)
R["norms"] = norm
R["norm_items_rev_mean_d"] = float(norm_items[norm_items.reverse].d.mean())
R["norm_items_fwd_mean_d"] = float(norm_items[~norm_items.reverse].d.mean())
# Holm-adjusted item comparisons
from statsmodels.stats.multitest import multipletests
norm_items["p_holm"] = multipletests(norm_items.p, method="holm")[1]
norm_items.to_csv(OUT / "norm_item_comparison.csv", index=False)
R["norm_items"] = norm_items.round(4).to_dict("records")

# ------------------------------------------------------------------ demographics (exploratory)
demo = {}
for k in ["Safe", "Enticing", "Alive", "Good", "Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
    a, b = s.loc[s.sex == 1, k], s.loc[s.sex == 0, k]
    t = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    r_age = stats.spearmanr(s[k], s.age)
    demo[k] = dict(M1=a.mean(), M0=b.mean(), g=(a.mean() - b.mean()) / sp, p=float(t.pvalue),
                   rho_age=float(r_age.statistic), p_age=float(r_age.pvalue))
R["demographics"] = demo

# ------------------------------------------------------------------ sensitivity: drop longstring-flagged
keep = ~df.flag_longstring.values
sens = {}
for k in ["Safe", "Enticing", "Alive", "Good", "Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
    sens[k] = dict(alpha=alpha(s.loc[keep, SC[k]]), M=s.loc[keep, k].mean())
for g, e in [("Safe", "Eng_Safe"), ("Enticing", "Eng_Enticing"), ("Alive", "Eng_Alive")]:
    dlt = (s.loc[keep, e] - s.loc[keep, g]); sens[f"dz_{e}"] = float(dlt.mean() / dlt.std())
    sens[f"r_{g}~{e}"] = float(s.loc[keep, g].corr(s.loc[keep, e]))
sens["r_ARS~Eng_Total"] = float(s.loc[keep, "ARS"].corr(s.loc[keep, "Eng_Total"]))
ev, thr, k = parallel(s.loc[keep, ENG]); sens["ENG_PA_k"] = k; sens["ENG_var1"] = float(ev[0] / 18)
sens["n"] = int(keep.sum())
R["sensitivity_drop_longstring"] = sens

# ------------------------------------------------------------------ reconciliation with course report
# report: N=115/116, items on raw -5..5 metric, no reverse scoring
R["raw_item_means_for_report_check"] = {c: float(df[c].mean()) for c in PI + ENG}

def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    return o
(OUT / "results.json").write_text(json.dumps(conv(R), indent=1, default=str))
s.to_csv(OUT / "scored.csv", index=False)
df.to_csv(OUT / "clean_with_quality.csv", index=False)
print("done")

# ================================================================== supplementary robustness block
S2 = {}
# 1. keying-matched comparison: forward-keyed general items only vs (all forward-keyed) engineering items
for g, items, e in [("Safe", SAFE, "Eng_Safe"), ("Enticing", ENTICING, "Eng_Enticing"), ("Alive", ALIVE, "Eng_Alive")]:
    f = [c for c in items if c not in REV]; gf = s[f].mean(axis=1); dlt = s[e] - gf
    t = stats.ttest_rel(s[e], gf)
    S2[f"keymatched_{g}"] = dict(items=f, M_gen_fwd=gf.mean(), M_eng=s[e].mean(), diff=dlt.mean(), CI=mean_ci(dlt),
                                 t=float(t.statistic), p=float(t.pvalue), dz=float(dlt.mean() / dlt.std()))
# 2. content-balanced acquiescence (Safe + Enticing: 3 reversed items each)
def ars_bal(subsets):
    parts = []
    for v in subsets:
        f = [c for c in v if c not in REV]; r_ = [c for c in v if c in REV]
        parts.append((df[f].mean(axis=1) + df[r_].mean(axis=1)) / 2)
    return sum(parts) / len(parts)
df["ARS_SE"] = ars_bal([SAFE, ENTICING]); s["ARS_SE"] = df["ARS_SE"]
S2["ARS_SE"] = dict(M=df.ARS_SE.mean(), SD=df.ARS_SE.std(), CI=mean_ci(df.ARS_SE),
                    t=float(stats.ttest_1samp(df.ARS_SE, 0).statistic), p=float(stats.ttest_1samp(df.ARS_SE, 0).pvalue),
                    d=float(df.ARS_SE.mean() / df.ARS_SE.std()), share_pos=float((df.ARS_SE > 0).mean()))
for k in ["Safe", "Enticing", "Alive", "Good", "Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
    r = stats.pearsonr(s["ARS_SE"], s[k]); S2[f"r_ARS_SE~{k}"] = dict(r=float(r.statistic), p=float(r.pvalue), CI=r_ci(r.statistic, len(s)))
for y in ["Eng_Safe", "Eng_Enticing", "Eng_Alive", "Eng_Total"]:
    a = ols(y, ["Safe", "Enticing", "Alive"], s); b = ols(y, ["Safe", "Enticing", "Alive", "ARS_SE"], s)
    S2[f"reg_{y}"] = dict(R2_gen=a["R2"], R2_gen_ARS=b["R2"], dR2=b["R2"] - a["R2"], beta_ARS=b["beta"][-1], p_ARS=b["p"][-1],
                          betas_gen=a["beta"], p_gen=a["p"][1:], F=a["F"], Fdf=a["df"], Fp=a["Fp"],
                          F_change=float(((b["R2"] - a["R2"]) / 1) / ((1 - b["R2"]) / (len(s) - 5))),
                          p_change=float(stats.f.sf(((b["R2"] - a["R2"])) / ((1 - b["R2"]) / (len(s) - 5)), 1, len(s) - 5)))
# 3. specificity of the domain elevation: is (EngSafe - Safe) larger than the Enticing / Alive gaps?
gS = s.Eng_Safe - s.Safe; gE = s.Eng_Enticing - s.Enticing; gA = s.Eng_Alive - s.Alive
for nm, other in [("Safe_vs_Enticing", gE), ("Safe_vs_Alive", gA)]:
    dd = gS - other; t = stats.ttest_rel(gS, other)
    S2[f"gap_{nm}"] = dict(diff=dd.mean(), CI=mean_ci(dd), t=float(t.statistic), p=float(t.pvalue), dz=float(dd.mean() / dd.std()))
# same with keying-matched general scores
gSf = s.Eng_Safe - s[["P_safe", "P_good", "P_harmless"]].mean(axis=1)
gEf = s.Eng_Enticing - s[["P_beautyugly", "P_abundant", "P_fascinating", "P_beautyaround"]].mean(axis=1)
gAf = s.Eng_Alive - s[["P_helpme", "P_meant", "P_needsme", "P_reason"]].mean(axis=1)
for nm, other in [("Safe_vs_Enticing_fwd", gEf), ("Safe_vs_Alive_fwd", gAf)]:
    dd = gSf - other; t = stats.ttest_rel(gSf, other)
    S2[f"gap_{nm}"] = dict(diff=dd.mean(), CI=mean_ci(dd), t=float(t.statistic), p=float(t.pvalue), dz=float(dd.mean() / dd.std()))
# 4. 2x2 cross-tab of general Safe vs engineering Safe (midpoint 2.5)
ct = pd.crosstab(s.Safe > 2.5, s.Eng_Safe > 2.5)
S2["safe_crosstab"] = {f"gen_{'hi' if a else 'lo'}|eng_{'hi' if b else 'lo'}": int(ct.loc[a, b]) for a in ct.index for b in ct.columns}
# 5. age and engineering security (exploratory)
r1 = stats.spearmanr(s.age, s.Eng_Safe); m = s.age <= 22; r2 = stats.spearmanr(s.age[m], s.Eng_Safe[m])
rp = stats.pearsonr(s.age, s.Eng_Safe)
S2["age_EngSafe"] = dict(rho=float(r1.statistic), p=float(r1.pvalue), rho_le22=float(r2.statistic), p_le22=float(r2.pvalue),
                         r=float(rp.statistic), r_CI=r_ci(rp.statistic, len(s)), n_le22=int(m.sum()),
                         means_by_age={int(a): float(v) for a, v in s.groupby("age").Eng_Safe.mean().items()})
for it in E_SAFE:
    r_ = stats.spearmanr(s.age, s[it]); S2[f"age_{it}"] = dict(rho=float(r_.statistic), p=float(r_.pvalue))
# 6. agreement with key single items (raw > 0)
for c in PI + ENG: S2[f"agree_{c}"] = float((df[c] > 0).mean())
S2["agree_cutthroat_strong"] = float((df["P_cutthroat"] >= 3).mean())
# 7. item-rest correlations of Safe items
S2["citc_Safe"] = R["citc"]["Safe"]
# 8. Holm correction across the 8 sex + 8 age exploratory tests
ps = [R["demographics"][k]["p"] for k in R["demographics"]] + [R["demographics"][k]["p_age"] for k in R["demographics"]]
S2["holm_demo"] = dict(zip([f"sex_{k}" for k in R["demographics"]] + [f"age_{k}" for k in R["demographics"]],
                           multipletests(ps, method="holm")[1].tolist()))
# 9. CFA on engineering items with a 2-factor (Security vs Engagement-Meaning) model
m_e2 = f"ESec =~ {' + '.join(E_SAFE)}\nEEng =~ {' + '.join(E_ENT + E_ALIVE)}"
try:
    r_, est = cfa(m_e2, d, "ENG_2f"); r_["SRMR"] = srmr(m_e2, d, ENG); r_["latent_corr"] = latent_corr(est)
    S2["cfa_ENG_2f"] = r_
    dchi = r_["chi2"] - R["cfa"]["ENG_3f"]["chi2"]; ddf = r_["df"] - R["cfa"]["ENG_3f"]["df"]
    S2["diff_ENG_2f_vs_3f"] = dict(dchi2=dchi, ddf=ddf, p=float(stats.chi2.sf(dchi, ddf)))
except Exception as e:
    S2["cfa_ENG_2f"] = dict(error=str(e))
# 10. raw-metric item means to compare with the course report (N=115) values
R["S2"] = S2
(OUT / "results.json").write_text(json.dumps(conv(R), indent=1, default=str))
s.to_csv(OUT / "scored.csv", index=False)
print("supplement done")
