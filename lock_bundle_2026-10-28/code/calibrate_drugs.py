"""
Calibration on FAKE trial outcomes (real outcomes are not read).
Question-A test: among programs whose target is not itself DNA-backed (distance >= 1), do targets 1 link
away "succeed" more than targets 2+ links away?  Statistic: relative success RS = P(s | d=1) / P(s | d>=2).
Null distributions:
  plain       shuffle success labels across all programs  -> count of successes at d=1 ~ hypergeometric
  fame-aware  shuffle only within the 5 fame groups      -> sum of hypergeometrics, one per group
Fake outcome sets (each 300 times, success rate about 15%):
  random     success independent of everything
  fame-only  success more likely for famous targets (odds x1.5 per standard deviation of log papers),
             with NO link to distance
  planted    success more likely at d=1 (odds x1.3 or x1.6), fame-neutral: to measure power
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(20261026)
F = pd.read_csv('/home/claude/atlas/drugs/program_features.tsv', sep='\t')
F = F[np.isfinite(F.d_full) & (F.d_full >= 1)].reset_index(drop=True)
near = (F.d_full == 1).to_numpy()
g = F.fame_group.to_numpy()
lf = np.log1p(F.papers_to2022.to_numpy()); lf = (lf - lf.mean()) / lf.std()
print(f'programs in the Question-A set: {len(F)}; at d=1: {near.sum()}')

def pvals(y, draws=10000):
    obs = y[near].sum(); K = y.sum(); Nn = len(y); n1 = near.sum()
    plain = rng.hypergeometric(K, Nn - K, n1, size=draws)
    strat = np.zeros(draws)
    for k in range(1, 6):
        sel = g == k
        Kg = y[sel].sum(); Ng = sel.sum(); n1g = (near & sel).sum()
        if n1g and Kg:
            strat += rng.hypergeometric(Kg, Ng - Kg, n1g, size=draws)
    return (1 + (plain >= obs).sum()) / (draws + 1), (1 + (strat >= obs).sum()) / (draws + 1)

def logistic(x):
    return 1 / (1 + np.exp(-x))
base = np.log(0.15 / 0.85)
scen = {
    'random':              lambda: rng.random(len(F)) < 0.15,
    'fame-only':           lambda: rng.random(len(F)) < logistic(base + np.log(1.5) * lf),
    'planted x1.3 (weak)':  lambda: rng.random(len(F)) < logistic(base + np.log(1.3) * near),
    'planted x1.6 (strong)': lambda: rng.random(len(F)) < logistic(base + np.log(1.6) * near),
}
rows = []
for name, gen in scen.items():
    for rep in range(300):
        y = gen()
        p_plain, p_fame = pvals(y)
        rs = y[near].mean() / y[~near].mean()
        rows.append(dict(fake=name, rs=rs, plain_sig=p_plain < 0.05, fame_sig=p_fame < 0.05))
R = pd.DataFrame(rows)
S = R.groupby('fake', sort=False).agg(mean_RS=('rs', 'mean'), plain_shuffle=('plain_sig', 'mean'), fame_aware_shuffle=('fame_sig', 'mean'))
print(S.round(3).to_string())
S.to_csv('/home/claude/atlas/drugs/calibration_drugs.tsv', sep='\t')
