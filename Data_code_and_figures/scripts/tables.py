"""Build formatted table content (JSON) for the manuscript and the online supplement."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"
R = json.loads((OUT / "results.json").read_text()); S2 = R["S2"]
s = pd.read_csv(OUT / "scored.csv")

def f2(x, lead=True):
    t = f"{x:.2f}"
    if not lead: t = t.replace("0.", ".", 1) if x >= 0 else t.replace("-0.", "−.", 1)
    return t.replace("-", "−")
def nl(x): return f2(x, lead=False)
def f3nl(x): t = f"{x:.3f}"; return (t.replace("0.", ".", 1) if x >= 0 else t.replace("-0.", "−.", 1)).replace("-", "−")
def ci(a, lead=True): return f"[{f2(a[0], lead)}, {f2(a[1], lead)}]"
def pfmt(p): return "< .001" if p < .001 else nl(p) if p >= .01 else f3nl(p)
def pct(x): return f"{100 * x:.1f}"

T = {}
# ---------------------------------------------------------------- Table 1 descriptives & reliability
rel, norm = R["reliability"], R["norms"]
rows = []
labels = [("Safe", "General Safe (PI-18)"), ("Enticing", "General Enticing (PI-18)"), ("Alive", "General Alive (PI-18)"),
          ("Good", "General Good (PI-18)"), ("Eng_Safe", "Engineering Safe (EPB-18)"), ("Eng_Enticing", "Engineering Enticing (EPB-18)"),
          ("Eng_Alive", "Engineering Alive (EPB-18)"), ("Eng_Total", "Engineering total (EPB-18)")]
for k, lab in labels:
    r = rel[k]
    ref = norm.get(k)
    rows.append([lab, str(r["k"]), f"{f2(r['M'])} {ci(r['M_CI'])}", f2(r["SD"]), f"{nl(r['alpha'])} {ci(r['alpha_CI'], False)}",
                 nl(r["omega"]), pct(r["share_above_mid"]),
                 f2(ref["norm_M"]) if ref else "—", f2(ref["d"]) if ref else "—"])
T["t1"] = dict(title="Descriptive Statistics and Reliability of General and Engineering-Specific Primal Belief Scales (N = 108)",
               header=["Scale", "k", "M [95% CI]", "SD", "α [95% CI]", "ω", "% > midpoint", "Ref. M", "d vs. ref."],
               widths=[2550, 380, 1500, 560, 1450, 560, 900, 700, 750], rows=rows,
               note=("Note. All scores range 0–5 (higher = more Safe, Enticing, Alive or Good); reverse-keyed PI-18 items were rescored. "
                     "α = Cronbach's alpha with 4,000-sample percentile bootstrap CI; ω = McDonald's omega total from a one-factor model. "
                     "% > midpoint = percentage of respondents scoring above 2.5, i.e., net agreement with the positive pole. "
                     "Ref. M = mean of the item means reported for Study 2 (N = 5,171) in the online supplement of Clifton and Yaden (2021); "
                     "d = (sample M − Ref. M) / sample SD. All four differences from the reference values were significant, smallest t(107) = −2.45, p = .016. "
                     "No reference values exist for the engineering-specific items."))

# ---------------------------------------------------------------- Table 2 measurement models
cfa = R["cfa"]; e2 = S2["cfa_ENG_2f"]
def mrow(lab, m, extra=""):
    return [lab, f"{m['chi2']:.2f} ({int(m['df'])})", f3nl(m["CFI"])[:-1] if False else f"{m['CFI']:.3f}".replace("0.", "."),
            f"{m['TLI']:.3f}".replace("0.", "."), f"{m['RMSEA']:.3f}".replace("0.", "."), f"{m['SRMR']:.3f}".replace("0.", "."), extra]
class LC(dict):
    def __getitem__(self, k):
        a, b = k.split("~")
        return dict.get(self, f"{a}~{b}", dict.get(self, f"{b}~{a}"))
lc = LC(cfa["PI_3f"]["latent_corr"]); lcm = LC(cfa["PI_3f_method"]["latent_corr"]); le = LC(cfa["ENG_3f"]["latent_corr"])
d1 = cfa["diff_PI_3f_vs_PI_3f_method"]; d2 = cfa["diff_ENG_1f_vs_ENG_3f"]; d3 = S2["diff_ENG_2f_vs_3f"]
rows = [["PI-18 (general primals)", "", "", "", "", "", ""],
        mrow("  One factor", cfa["PI_1f"]),
        mrow("  Three correlated factors (Safe, Enticing, Alive)", cfa["PI_3f"],
             f"φ: S–E = {nl(lc['Safe~Enticing'])}, S–A = {nl(lc['Safe~Alive'])}, E–A = {nl(lc['Enticing~Alive'])}"),
        mrow("  Three factors + orthogonal reverse-keying factor", cfa["PI_3f_method"],
             f"Δχ²(7) = {d1['dchi2']:.2f}, p {pfmt(d1['p'])} vs. three factors"),
        ["EPB-18 (engineering-specific)", "", "", "", "", "", ""],
        mrow("  One factor", cfa["ENG_1f"]),
        mrow("  Two factors (Security; Engagement–Meaning)", e2, f"φ = {nl(LC(e2['latent_corr'])['EEng~ESec'])}"),
        mrow("  Three correlated factors (Safe, Enticing, Alive)", cfa["ENG_3f"],
             f"φ: S–E = {nl(le['ESafe~EEnt'])}, S–A = {nl(le['ESafe~EAlive'])}, E–A = {nl(le['EAlive~EEnt'])}; "
             f"Δχ²(2) = {d3['dchi2']:.2f} vs. two factors, Δχ²(3) = {d2['dchi2']:.2f} vs. one factor, both p < .001")]
T["t2"] = dict(title="Fit of Confirmatory Measurement Models for the General (PI-18) and Engineering-Specific (EPB-18) Items",
               header=["Model", "χ² (df)", "CFI", "TLI", "RMSEA", "SRMR", "Latent correlations / comparison"],
               widths=[3100, 1150, 600, 600, 750, 700, 2400], rows=rows,
               note=("Note. Maximum-likelihood estimation (semopy 2.3.11), N = 108; all χ² tests p < .001. φ = standardised latent correlation; "
                     "S = Safe, E = Enticing, A = Alive. The reverse-keying factor was specified on the seven reverse-keyed PI-18 items and constrained "
                     "to be uncorrelated with the substantive factors. Conventional benchmarks (CFI/TLI ≥ .95, RMSEA ≤ .06, SRMR ≤ .08; Hu & Bentler, 1999) "
                     "are stringent for N ≈ 100 and 18 indicators, so fit is interpreted comparatively."))

# ---------------------------------------------------------------- Table 3 correlations
V = [("Safe", "1. General Safe"), ("Enticing", "2. General Enticing"), ("Alive", "3. General Alive"), ("Good", "4. General Good"),
     ("Eng_Safe", "5. Engineering Safe"), ("Eng_Enticing", "6. Engineering Enticing"), ("Eng_Alive", "7. Engineering Alive"),
     ("Eng_Total", "8. Engineering total"), ("ARS_SE", "9. Acquiescence index"), ("age", "10. Age")]
C = s[[v for v, _ in V]].corr()
rows = []
for i, (v, lab) in enumerate(V):
    row = [lab]
    for j, (w, _) in enumerate(V):
        if j < i:
            r = C.loc[v, w]; p = stats.pearsonr(s[v], s[w])[1]
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            row.append(nl(r) + star)
        elif j == i: row.append("—")
        else: row.append("")
    rows.append(row[:-1])
T["t3"] = dict(title="Pearson Correlations Among General and Engineering-Specific Primal Beliefs, Acquiescence and Age",
               header=["Variable"] + [str(i + 1) for i in range(9)], widths=[2600] + [780] * 9, rows=rows,
               note=("Note. N = 108. Correlations between overlapping composites (e.g., Good with Safe and Enticing; engineering total with its subscales) "
                     "are inflated by shared items. Acquiescence index = content-balanced mean of raw responses to forward- and reverse-keyed PI-18 Safe and "
                     "Enticing items (see Method). Matched cross-domain correlations with 5,000-sample bootstrap 95% CIs: Safe = .30 [.13, .47], "
                     "Enticing = .34 [.14, .52], Alive = .35 [.16, .52]. *p < .05. **p < .01. ***p < .001."))

# ---------------------------------------------------------------- Table 4 domain vs general
dv = R["domain_vs_general"]
rows = []
for g, e, lab in [("Safe", "Eng_Safe", "Safe"), ("Enticing", "Eng_Enticing", "Enticing"), ("Alive", "Eng_Alive", "Alive")]:
    c = dv[f"{e}-{g}"]; km = S2[f"keymatched_{g}"]
    rows.append([lab, f"{f2(c['M_gen'])} ({f2(c['SD_gen'])})", f"{f2(c['M_eng'])} ({f2(c['SD_eng'])})",
                 f"{f2(c['diff'])} {ci(c['diff_CI'])}", f"{f2(c['t'])}", pfmt(c["p"]), f"{f2(c['dz'])} {ci(c['dz_CI'])}",
                 f"{f2(km['diff'])} {ci(km['CI'])}", f"{f2(km['dz'])}"])
T["t4"] = dict(title="Engineering-Specific Versus General Primal Beliefs: Within-Person Comparisons (N = 108)",
               header=["Primal", "General M (SD)", "Engineering M (SD)", "Difference [95% CI]", "t(107)", "p", "dz [95% CI]",
                       "Keying-matched difference [95% CI]", "Keying-matched dz"],
               widths=[780, 1080, 1180, 1720, 760, 720, 1680, 1720, 900], rows=rows,
               note=("Note. Paired-samples t tests; dz = mean difference / SD of differences, with 5,000-sample percentile bootstrap CI. "
                     "Wilcoxon signed-rank tests gave the same pattern (Safe p < .001; Enticing p = .033; Alive p = .029). "
                     f"Percentage of students scoring higher on the engineering scale: Safe {pct(dv['Eng_Safe-Safe']['share_eng_higher'])}, "
                     f"Enticing {pct(dv['Eng_Enticing-Enticing']['share_eng_higher'])}, Alive {pct(dv['Eng_Alive-Alive']['share_eng_higher'])}. "
                     "Keying-matched comparisons use only the positively keyed PI-18 items of each primal (Safe: 3 items; Enticing: 4; Alive: 4), "
                     "because all EPB-18 items are positively keyed. The Safe gap exceeded the Enticing gap by "
                     f"{f2(S2['gap_Safe_vs_Enticing']['diff'])} {ci(S2['gap_Safe_vs_Enticing']['CI'])}, dz = {f2(S2['gap_Safe_vs_Enticing']['dz'])}, "
                     f"and the Alive gap by {f2(S2['gap_Safe_vs_Alive']['diff'])} {ci(S2['gap_Safe_vs_Alive']['CI'])}, dz = {f2(S2['gap_Safe_vs_Alive']['dz'])}, both p < .001."))

# ---------------------------------------------------------------- Table 5 regressions
rows = []
for y, lab in [("Eng_Safe", "Engineering Safe"), ("Eng_Enticing", "Engineering Enticing"), ("Eng_Alive", "Engineering Alive"), ("Eng_Total", "Engineering total")]:
    g = R["regression"][y]["general_only"]; q = S2[f"reg_{y}"]
    b = q["betas_gen"]; p = q["p_gen"]
    def bs(i): return nl(b[i]) + ("*" if p[i] < .05 else "")
    rows.append([lab, bs(0), bs(1), bs(2), f"{nl(g['R2'])} {ci(g['R2_CI'], False)}",
                 nl(q["beta_ARS"]) + ("**" if q["p_ARS"] < .01 else "*" if q["p_ARS"] < .05 else ""),
                 f"{q['dR2']:.3f}".replace("0.", "."), pfmt(q["p_change"])])
T["t5"] = dict(title="Hierarchical Regressions Predicting Engineering-Specific Beliefs From General Primals (Step 1) and Acquiescence (Step 2)",
               header=["Outcome", "β Safe", "β Enticing", "β Alive", "Step 1 R² [95% CI]", "β Acquiescence (Step 2)", "ΔR²", "p (ΔR²)"],
               widths=[2000, 900, 1050, 900, 1700, 1500, 800, 900], rows=rows,
               note=("Note. N = 108. β = standardised coefficients from Step 1 (general primals only). R² CIs are 2,000-sample percentile bootstrap intervals. "
                     "Step 1 models were all significant, smallest F(3, 104) = 4.86, p = .003. *p < .05. **p < .01."))

# ---------------------------------------------------------------- Supplement tables
items = pd.read_csv(OUT / "item_descriptives.csv").set_index("item")
L3 = pd.read_csv(OUT / "efa_ENG18_3f.csv", index_col=0)
ENGTXT = {
 "ES1": "I feel confident that my engineering degree will provide a secure future.",
 "ES2": "The engineering environment feels safe and supportive.",
 "ES3": "Most challenges in my engineering coursework can be solved with effort and perseverance.",
 "ES4": "I trust that my education is preparing me for a stable career in engineering.",
 "ES5": "My engineering program offers a solid foundation for a reliable future.",
 "ES6": "The academic structure in my engineering program gives me a sense of security.",
 "EE1": "Engineering offers endless opportunities for innovation and creativity.",
 "EE2": "I find the problems we solve in engineering to be exciting and engaging.",
 "EE3": "There is always something new to explore in the field of engineering.",
 "EE4": "The complexity of engineering makes it fascinating to study and work in.",
 "EE5": "I feel energized by the possibilities that engineering brings to the world.",
 "EE6": "The engineering projects I work on are full of potential and discovery.",
 "EA1": "My engineering studies feel purposeful and meaningful.",
 "EA2": "The projects I work on seem to follow a greater goal.",
 "EA3": "I believe my role as an engineer is important for solving real-world problems.",
 "EA4": "I often feel like my education is preparing me for a larger purpose.",
 "EA5": "The challenges I face in engineering help me grow in meaningful ways.",
 "EA6": "The work I do in engineering feels connected to something bigger than myself."}
citc = {}
for k, v in [("Eng_Safe", ["ES1", "ES2", "ES3", "ES4", "ES5", "ES6"]), ("Eng_Enticing", [f"EE{i}" for i in range(1, 7)]), ("Eng_Alive", [f"EA{i}" for i in range(1, 7)])]:
    citc.update(R["citc"][k])
rows = []
for it, txt in ENGTXT.items():
    r = items.loc[it]
    rows.append([it, txt, f2(r.M), f2(r.SD), pct(r.pct_agree), nl(citc[it]), nl(L3.loc[it, "F1"]), nl(L3.loc[it, "F2"]), nl(L3.loc[it, "F3"])])
T["s1"] = dict(title="Engineering-Specific Primal Belief Items (EPB-18): Wording, Descriptive Statistics and Exploratory Factor Loadings",
               header=["Code", "Item", "M", "SD", "% agree", "r(it)", "F1 Safe", "F2 Enticing", "F3 Alive"],
               widths=[600, 4400, 550, 550, 750, 600, 700, 800, 700], rows=rows,
               note=("Note. N = 108. Items were answered on a six-point scale (strongly disagree to strongly agree) and scored 0–5. "
                     "% agree = percentage choosing slightly agree, agree or strongly agree. r(it) = corrected item–rest correlation within the intended subscale. "
                     "Loadings are pattern coefficients from minimum-residual EFA with oblimin rotation (three factors); |λ| ≥ .40 is the conventional salience threshold. "
                     "Factor correlations: F1–F2 = .36, F1–F3 = .60, F2–F3 = .60. Item codes: ES = Safe, EE = Enticing, EA = Alive."))

ni = pd.read_csv(OUT / "norm_item_comparison.csv").set_index("item")
PITXT = {"P_safe": "I tend to see the world as pretty safe.", "P_dangerous": "On the whole, the world is a dangerous place. (R)",
         "P_cutthroat": "Instead of being cooperative, the world is a cut-throat and competitive place. (R)", "P_worse": "Most things have a habit of getting worse. (R)",
         "P_good": "Most things in the world are good.", "P_harmless": "Most things and situations are harmless and totally safe.",
         "P_beautyugly": "In life, there's way more beauty than ugliness.", "P_worth": "While some things are worth checking out or exploring further, most things probably aren't worth the effort. (R)",
         "P_boring": "Most things in life are kind of boring. (R)", "P_abundant": "The world is an abundant place with tons and tons to offer.",
         "P_fascinating": "No matter where we are or what the topic might be, the world is fascinating.", "P_dull": "The world is a somewhat dull place where plenty of things are not that interesting. (R)",
         "P_beautyaround": "No matter where we are, incredible beauty is always around us.", "P_helpme": "It often feels like events are happening in order to help me in some way.",
         "P_meant": "What happens in the world is meant to happen.", "P_lackpurpose": "Events seem to lack any cosmic or bigger purpose. (R)",
         "P_needsme": "The universe needs me for something important.", "P_reason": "Everything happens for a reason and on purpose."}
GRP = [("Safe", ["P_safe", "P_dangerous", "P_cutthroat", "P_worse", "P_good", "P_harmless"]),
       ("Enticing", ["P_beautyugly", "P_worth", "P_boring", "P_abundant", "P_fascinating", "P_dull", "P_beautyaround"]),
       ("Alive", ["P_helpme", "P_meant", "P_lackpurpose", "P_needsme", "P_reason"])]
L3p = pd.read_csv(OUT / "efa_PI18_3f.csv", index_col=0)
cfa_l = R["cfa"]["PI_3f"]["std_loadings"]; cfam = R["cfa"]["PI_3f_method"]["std_loadings"]
SHORT = {"P_safe": "World is pretty safe", "P_dangerous": "World is dangerous (R)", "P_cutthroat": "World is cut-throat, competitive (R)",
         "P_worse": "Things tend to get worse (R)", "P_good": "Most things are good", "P_harmless": "Most things harmless, totally safe",
         "P_beautyugly": "More beauty than ugliness", "P_worth": "Most things not worth the effort (R)", "P_boring": "Most things kind of boring (R)",
         "P_abundant": "World is abundant", "P_fascinating": "World is fascinating", "P_dull": "World is somewhat dull (R)",
         "P_beautyaround": "Incredible beauty always around us", "P_helpme": "Events happen to help me", "P_meant": "What happens is meant to happen",
         "P_lackpurpose": "Events lack bigger purpose (R)", "P_needsme": "Universe needs me", "P_reason": "Everything happens for a reason"}
rows = []
for g, its in GRP:
    rows.append([g] + [""] * 11)
    for it in its:
        r = ni.loc[it]
        meth = cfam.get(f"Method->{it}")
        rows.append(["  " + SHORT[it], f2(r.M), f2(r.SD), pct(items.loc[it, "pct_agree"]), f2(r.norm_M), f2(r.d), pfmt(r.p_holm),
                     nl(L3p.loc[it, "F3"]), nl(L3p.loc[it, "F1"]), nl(L3p.loc[it, "F2"]),
                     nl(cfa_l[f"{g}->{it}"]), nl(meth) if meth is not None else "—"])
T["s2"] = dict(title="PI-18 Items: Descriptive Statistics, Comparison With Reference Means, and Factor Loadings",
               header=["Item", "M", "SD", "% agree", "Ref. M", "d", "p (Holm)", "EFA Safe", "EFA Ent.", "EFA Alive", "CFA λ", "λ method"],
               widths=[2900, 520, 520, 650, 620, 600, 700, 650, 650, 650, 600, 650], rows=rows,
               note=("Note. N = 108. Short item labels are given; full wording is published in Clifton and Yaden (2021). Scores 0–5 after reverse-scoring items marked (R); "
                     "% agree refers to the raw (unreversed) statement. Ref. M and the reference SDs used for d are from Study 2 (N = 5,171) of Clifton and Yaden (2021, online supplement, Table S2); "
                     "d uses the pooled SD of the two samples. p (Holm) = one-sample t test against the reference mean, Holm-adjusted across 18 items. "
                     "EFA = pattern coefficients from a three-factor minimum-residual EFA with oblimin rotation (factor correlations .12–.33). "
                     "CFA λ = standardised loading in the three-factor CFA; λ method = loading on the orthogonal reverse-keying factor in the extended model."))

sens = R["sensitivity_drop_longstring"]
rows = []
for k, lab in labels:
    rows.append([lab, f"{nl(rel[k]['alpha'])} / {nl(sens[k]['alpha'])}", f"{f2(rel[k]['M'])} / {f2(sens[k]['M'])}"])
rows += [["dz Engineering − General Safe", "", f"{f2(dv['Eng_Safe-Safe']['dz'])} / {f2(sens['dz_Eng_Safe'])}"],
         ["dz Engineering − General Enticing", "", f"{f2(dv['Eng_Enticing-Enticing']['dz'])} / {f2(sens['dz_Eng_Enticing'])}"],
         ["dz Engineering − General Alive", "", f"{f2(dv['Eng_Alive-Alive']['dz'])} / {f2(sens['dz_Eng_Alive'])}"],
         ["r General Safe – Engineering Safe", "", f"{nl(R['cross']['Safe~Eng_Safe']['r'])} / {nl(sens['r_Safe~Eng_Safe'])}"],
         ["Parallel analysis: EPB-18 factors retained", "", f"{R['efa']['ENG18']['n_factors_PA']} / {sens['ENG_PA_k']}"],
         ["EPB-18 first-eigenvalue variance share", "", f"{pct(R['efa']['ENG18']['var1'])}% / {pct(sens['ENG_var1'])}%"]]
T["s3"] = dict(title="Sensitivity Analysis: Full Sample (N = 108) Versus Sample Excluding Six Long-String Responders (n = 102)",
               header=["Quantity", "α (full / reduced)", "Estimate (full / reduced)"], widths=[4600, 2200, 2400], rows=rows,
               note=("Note. Long-string responders gave the identical response to 18 consecutive items (spanning the EPB-18 block in all six cases). "
                     "Four of the six answered every engineering item positively and two answered every engineering item negatively; such uniform answers may be genuine, so exclusion is reported only as a sensitivity check."))
(OUT / "tables.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
print("tables ok")

# ---------------------------------------------------------------- APA italics for statistical symbols (headers, notes)
import re
def ital(x):
    x = re.sub(r"\bdz\b", "*d*~z~", x)
    x = re.sub(r"(?<![\w*])N = ", "*N* = ", x); x = re.sub(r"(?<![\w*])n = ", "*n* = ", x)
    x = re.sub(r"(?<![\w*])p ([<=≤])", r"*p* \1", x); x = re.sub(r"(?<![\w*])p \(", "*p* (", x)
    x = re.sub(r"(?<![\w*])t\((\d+)\)", r"*t*(\1)", x); x = re.sub(r"(?<![\w*])F\((\d)", r"*F*(\1", x)
    x = re.sub(r"(?<![\w*])d = ", "*d* = ", x); x = re.sub(r"(?<![\w*])d uses", "*d* uses", x); x = re.sub(r"(?<![\w*])d vs\.", "*d* vs.", x)
    x = x.replace("ΔR²", "Δ*R*²"); x = re.sub(r"(?<![\w*])R²", "*R*²", x); x = x.replace("χ² (df)", "χ² (*df*)"); x = re.sub(r"(?<![\w*])r\(it\)", "*r*(it)", x)
    return x
for k, t in T.items():
    t["note"] = ital(t["note"]).replace("***p < .001", "\\*\\*\\**p* < .001").replace("**p < .01", "\\*\\**p* < .01").replace("*p < .05", "\\**p* < .05")
    newh = []
    for h in t["header"]:
        h = ital(h)
        h = re.sub(r"^M\b", "*M*", h); h = re.sub(r"(?<=\s)M\b", "*M*", h); h = re.sub(r"\bSD\b", "*SD*", h)
        h = re.sub(r"^k$", "*k*", h); h = re.sub(r"^d$", "*d*", h); h = re.sub(r"^t\(", "*t*(", h); h = re.sub(r"^p$", "*p*", h)
        newh.append(h)
    t["header"] = newh
(OUT / "tables.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
print("italics ok")
