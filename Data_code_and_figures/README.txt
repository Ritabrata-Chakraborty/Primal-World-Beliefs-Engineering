Supplementary data, code and figures
"A Safe Harbour in a Cut-Throat World? General and Engineering-Specific Primal World Beliefs Among Indian Engineering Undergraduates"

data/pwb_engineering_deidentified.csv  De-identified responses (N = 108). Codebook: Online Supplement, section S3.
scripts/analysis.py   All statistics. Run:  python scripts/analysis.py data/pwb_engineering_deidentified.csv out
                      (writes out/results.json; fixed bootstrap seed 20241013 reproduces the reported values)
scripts/tables.py     Formats tables from out/results.json (expects scripts/ and out/ side by side)
scripts/figures.py    Regenerates the four figures from out/ (uses the Carlito font if installed)
figures/              Figures 1-4 as 600-dpi PNG files, numbered as in the manuscript

Requirements: Python 3.11+; pandas, numpy, scipy, statsmodels, matplotlib, factor_analyzer 0.5.1, semopy 2.3.11.
Note: factor_analyzer 0.5.1 calls a scikit-learn argument renamed in recent versions (force_all_finite -> ensure_all_finite);
use scikit-learn < 1.6 or apply that one-word rename.
