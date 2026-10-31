"""
Part 2 - History test on real drug-program outcomes up to Dec 2022.
Source: Minikel 2024 pp.tsv (Citeline Pharmaprojects, pulled 22 Dec 2022). 2023-2026 outcomes are NOT used.
Plan frozen 2026-10-26 22:35 ET (Agasya's notes + "defaults"; see ATLAS_BUILD_LOG.md).

Questions (from Agasya's notes):
  Q0  replication: target is itself DNA-backed (d=0) vs not  -> relative success (Minikel: about 2.6x)
  Q1  main: among targets that are NOT DNA-backed, do targets 1 link away succeed more than targets 2+ away,
      once fame is held equal (20 fame groups by papers per target up to 2022)?
  Q2  which links carry it: Q1 on the blind-screen half vs the literature-only half
  Q3  does fame predict success at the same DNA evidence and distance?
  Cross-checks: each protein counts once (weights); logistic regression with errors grouped by target.
  Secondary: the standard network score (Guney z) among d >= 1.

Definitions (Minikel): P(S) at a phase = share of programs with a known outcome at that phase that advanced.
I->Launch = product of P(S) for phases I, II, III.  Relative success RS = P(S | yes) / P(S | no).
Fame-adjusted RS = Mantel-Haenszel risk ratio across the 20 fame groups, per phase; I->Launch = product.
p-values: success labels shuffled only within fame groups (exact hypergeometric draws), each phase separately.
95% intervals: bootstrap over target proteins (Poisson weights), so programs sharing a target move together.
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import time, sys
import numpy as np, pandas as pd
import statsmodels.api as sm

t0 = time.time()
rng = np.random.default_rng(20261027)
D = '/home/claude/ext/ericminikel_genetic_support/data/'
OUT = '/home/claude/atlas/drugs/'
NG = 20          # fame groups (decided from fake-outcome calibration, Oct 26)
DRAWS = 20000    # shuffles per test
BOOT = 2000      # bootstrap resamples of targets

F = pd.read_csv(OUT + 'program_features.tsv', sep='\t')
pp = pd.read_csv(D + 'pp.tsv', sep='\t', usecols=['ti_uid', 'gene', 'indication_mesh_id', 'ccat', 'succ_p_1', 'succ_1_2', 'succ_2_3', 'succ_3_a'])
# pp.tsv has 16 repeated target-disease IDs (32 rows) with different histories; Minikel counts each row,
# so rows are joined by position (same row order and filter as part1_drugs.py), not by ID.
pp = pp.dropna(subset=['gene', 'indication_mesh_id']).reset_index(drop=True)
assert len(pp) == len(F) and (pp.ti_uid.to_numpy() == F.ti_uid.to_numpy()).all()
F = pd.concat([F, pp[['ccat', 'succ_p_1', 'succ_1_2', 'succ_2_3', 'succ_3_a']]], axis=1)
ind = pd.read_csv(D + 'indic.tsv', sep='\t', usecols=['indication_mesh_id', 'areas'])
F = F.merge(ind, on='indication_mesh_id', how='left')
F['areas'] = F.areas.fillna('other')
PH = [('I', 'succ_1_2'), ('II', 'succ_2_3'), ('III', 'succ_3_a')]
for _, c in PH + [('Preclinical', 'succ_p_1')]:
    F[c] = F[c].map({True: 1.0, False: 0.0, 'TRUE': 1.0, 'FALSE': 0.0})
# Minikel's published 2.6x is a count ratio: launched / programs with a known phase I outcome (still-active
# programs count as not launched). Kept as a second overall measure next to the pre-specified product.
F['launch_I'] = np.where(F.succ_1_2.notna(), (F.ccat == 'Launched').astype(float), np.nan)
F['tid'] = F.gene.astype('category').cat.codes.astype(int)
NT = F.tid.max() + 1
F['lp'] = np.log10(F.papers_to2022 + 1)
print(f'programs {len(F)}; targets {NT}; with full-network distance {np.isfinite(F.d_full).sum()}')

def fame_groups(papers, k=NG):
    return pd.qcut(papers, k, labels=False, duplicates='drop').astype(int)

# ---------------------------------------------------------------- core statistics
def mh(a, n1, c, n0):
    """Mantel-Haenszel risk ratio; inputs are arrays over groups (last axis)."""
    N = n1 + n0
    with np.errstate(invalid='ignore', divide='ignore'):
        num = np.where(N > 0, a * n0 / np.where(N > 0, N, 1), 0).sum(-1)
        den = np.where(N > 0, c * n1 / np.where(N > 0, N, 1), 0).sum(-1)
        return num / den

def tables(s, e, g, w, ng):
    a = np.bincount(g, weights=w * s * e, minlength=ng)
    n1 = np.bincount(g, weights=w * e, minlength=ng)
    c = np.bincount(g, weights=w * s * (1 - e), minlength=ng)
    n0 = np.bincount(g, weights=w * (1 - e), minlength=ng)
    return a, n1, c, n0

def analyse(df, expo, grp, label, weights=None, draws=DRAWS, boot=BOOT, phases=PH, quiet=False, seed=20261027):
    """Plain and group-adjusted RS per phase + I->Launch, shuffle p-values, target bootstrap CIs."""
    rng = np.random.default_rng(seed)   # fixed per test -> reproducible; same target resamples across tests
    ng = int(grp.max()) + 1
    res = {}; null_plain = np.ones(draws); null_adj = np.ones(draws)
    boot_plain = np.ones(boot); boot_adj = np.ones(boot)
    W = rng.poisson(1.0, size=(boot, NT)).astype(float)
    obs_plain_IL = 1.0; obs_adj_IL = 1.0
    for ph, col in phases:
        k = df[col].notna().to_numpy()
        s = df[col].to_numpy()[k]; e = expo[k].astype(float); g = grp[k]; t = df.tid.to_numpy()[k]
        w = np.ones(k.sum()) if weights is None else weights(df[k], e)
        a, n1, c, n0 = tables(s, e, g, w, ng)
        p1 = a.sum() / n1.sum(); p0 = c.sum() / n0.sum()
        rs_plain = p1 / p0; rs_adj = mh(a, n1, c, n0)
        # shuffles (unweighted only): hypergeometric within groups / across all
        if weights is None:
            K = (a + c).astype(int); Ng = (n1 + n0).astype(int); n1i = n1.astype(int)
            A = np.zeros((draws, ng))
            for j in range(ng):
                if n1i[j] > 0 and Ng[j] > n1i[j] and K[j] > 0:
                    A[:, j] = rng.hypergeometric(K[j], Ng[j] - K[j], n1i[j], size=draws)
                else:
                    A[:, j] = a[j]
            null_adj *= mh(A, n1, K - A, n0)
            Kt, Nt, n1t = int(K.sum()), int(Ng.sum()), int(n1i.sum())
            Ap = rng.hypergeometric(Kt, Nt - Kt, n1t, size=draws)
            null_plain *= (Ap / n1t) / ((Kt - Ap) / (Nt - n1t))
        # bootstrap over targets
        key = g * 4 + e.astype(int) * 2 + s.astype(int)
        M = np.zeros((NT, ng * 4)); np.add.at(M, (t, key), w)
        C = W @ M
        C = C.reshape(boot, ng, 4)  # [e0s0, e0s1, e1s0, e1s1]
        ba, bn1 = C[:, :, 3], C[:, :, 2] + C[:, :, 3]
        bc, bn0 = C[:, :, 1], C[:, :, 0] + C[:, :, 1]
        bp = (ba.sum(1) / bn1.sum(1)) / (bc.sum(1) / bn0.sum(1)); bm = mh(ba, bn1, bc, bn0)
        boot_plain *= bp; boot_adj *= bm
        obs_plain_IL *= rs_plain; obs_adj_IL *= rs_adj
        res[ph] = dict(n_yes=int(round(n1.sum())) if weights is None else round(n1.sum(), 1),
                       n_no=int(round(n0.sum())) if weights is None else round(n0.sum(), 1),
                       ps_yes=p1, ps_no=p0, rs_plain=rs_plain, rs_plain_l=np.nanpercentile(bp, 2.5), rs_plain_u=np.nanpercentile(bp, 97.5),
                       rs_fame=rs_adj, rs_fame_l=np.nanpercentile(bm, 2.5), rs_fame_u=np.nanpercentile(bm, 97.5))
    r = dict(n_yes=np.nan, n_no=np.nan, ps_yes=np.prod([res[p]['ps_yes'] for p, _ in phases]),
             ps_no=np.prod([res[p]['ps_no'] for p, _ in phases]),
             rs_plain=obs_plain_IL, rs_plain_l=np.nanpercentile(boot_plain, 2.5), rs_plain_u=np.nanpercentile(boot_plain, 97.5),
             rs_fame=obs_adj_IL, rs_fame_l=np.nanpercentile(boot_adj, 2.5), rs_fame_u=np.nanpercentile(boot_adj, 97.5))
    if weights is None:
        for nm, nul, ob in (('plain', null_plain, obs_plain_IL), ('fame', null_adj, obs_adj_IL)):
            hi = (1 + (nul >= ob).sum()) / (draws + 1); lo = (1 + (nul <= ob).sum()) / (draws + 1)
            r[f'p_{nm}_higher'] = hi; r[f'p_{nm}_two'] = min(1.0, 2 * min(hi, lo))
    res['I-Launch'] = r
    out = pd.DataFrame(res).T; out.insert(0, 'phase', out.index); out.insert(0, 'test', label)
    if not quiet:
        cols = ['n_yes', 'n_no', 'ps_yes', 'ps_no', 'rs_plain', 'rs_plain_l', 'rs_plain_u', 'rs_fame', 'rs_fame_l', 'rs_fame_u']
        cols += [c for c in ('p_plain_higher', 'p_fame_higher', 'p_fame_two') if c in out]
        print(f'\n== {label}  ({time.time()-t0:.0f}s)')
        print(out[cols].astype(float).round(4).to_string())
    return out, (boot_plain, boot_adj)

results = []
_analyse = analyse
def analyse(df, expo, grp, label, weights=None, **kw):
    out, boots = _analyse(df, expo, grp, label, weights=weights, **kw)
    if kw.get('phases') is None and not kw.get('quiet'):
        o2, _ = _analyse(df, expo, grp, label, weights=weights, phases=[('count', 'launch_I')], quiet=True)
        row = o2.loc['I-Launch'].copy(); row['phase'] = 'I-Launch (count, Minikel 2.6x style)'
        out = pd.concat([out, row.to_frame().T]); out.index = list(out.index[:-1]) + ['I-Launch count']
        r = row
        ptxt = f", p(fame-aware, two-sided)={r['p_fame_two']:.4f}" if 'p_fame_two' in r and pd.notna(r.get('p_fame_two')) else ''
        print(f"   overall as launched/entered phase I: plain {r['rs_plain']:.3f} ({r['rs_plain_l']:.3f}-{r['rs_plain_u']:.3f}), "
              f"fame-adjusted {r['rs_fame']:.3f} ({r['rs_fame_l']:.3f}-{r['rs_fame_u']:.3f}){ptxt}")
    return out, boots

# ---------------------------------------------------------------- Q0 replication: d = 0 vs not
S0 = F[np.isfinite(F.d_full)].reset_index(drop=True)
g0 = fame_groups(S0.papers_to2022).to_numpy()
o, _ = analyse(S0, (S0.d_full == 0).to_numpy(), g0, 'Q0 DNA-backed target (d=0) vs not, full network')
results.append(o)
# same, restricted to Minikel's own list of diseases with genetic insight (his require_insight rule)
ins = pd.read_csv(D + 'indic.tsv', sep='\t', usecols=['indication_mesh_id', 'genetic_insight'])
ins = set(ins.indication_mesh_id[ins.genetic_insight != 'none'])
S0m = S0[S0.indication_mesh_id.isin(ins)].reset_index(drop=True)
o, _ = analyse(S0m, (S0m.d_full == 0).to_numpy(), fame_groups(S0m.papers_to2022).to_numpy(),
               "Q0 replication on Minikel's genetic-insight diseases only")
results.append(o)

# ---------------------------------------------------------------- Q1 main: d = 1 vs d >= 2
S1 = F[np.isfinite(F.d_full) & (F.d_full >= 1)].reset_index(drop=True)
g1 = fame_groups(S1.papers_to2022).to_numpy()
print('\nfame groups (Q1 set): programs per group', np.bincount(g1).min(), '-', np.bincount(g1).max())
o, _ = analyse(S1, (S1.d_full == 1).to_numpy(), g1, 'Q1 MAIN: 1 link vs 2+ links (not DNA-backed), full network')
results.append(o)

# ---------------------------------------------------------------- Q2 halves
for net in ('systematic', 'literature'):
    S = F[np.isfinite(F[f'd_{net}']) & (F[f'd_{net}'] >= 1)].reset_index(drop=True)
    g = fame_groups(S.papers_to2022).to_numpy()
    o, _ = analyse(S, (S[f'd_{net}'] == 1).to_numpy(), g, f'Q2 1 vs 2+ links, {net} half (all programs with a distance there)')
    results.append(o)
# matched set: same programs in both halves; difference with a shared bootstrap
SM = F[np.isfinite(F.d_systematic) & np.isfinite(F.d_literature) & (F.d_systematic >= 1) & (F.d_literature >= 1)].reset_index(drop=True)
gm = fame_groups(SM.papers_to2022).to_numpy()
# both halves use the same fixed seed -> the same target resamples -> paired difference
o_s, (bp_s, bm_s) = analyse(SM, (SM.d_systematic == 1).to_numpy(), gm, 'Q2 matched programs: blind-screen half')
o_l, (bp_l, bm_l) = analyse(SM, (SM.d_literature == 1).to_numpy(), gm, 'Q2 matched programs: literature half')
results += [o_s, o_l]
ratio_obs = o_s.loc['I-Launch', 'rs_fame'] / o_l.loc['I-Launch', 'rs_fame']
rb = bm_s / bm_l
print(f'\nQ2 paired: fame-adjusted I-Launch RS blind-screen / literature = {ratio_obs:.3f} '
      f'(95% CI {np.nanpercentile(rb, 2.5):.3f}-{np.nanpercentile(rb, 97.5):.3f})')
q2pair = dict(test='Q2 paired ratio blind-screen/literature (fame-adjusted I-Launch)', phase='I-Launch',
              rs_fame=ratio_obs, rs_fame_l=np.nanpercentile(rb, 2.5), rs_fame_u=np.nanpercentile(rb, 97.5))
overlap = pd.crosstab(SM.d_systematic == 1, SM.d_literature == 1)
print('matched set: rows = 1 link in blind-screen half, cols = 1 link in literature half\n', overlap.to_string())

# ---------------------------------------------------------------- Q3 fame as a predictor
# more-famous half vs less-famous half, compared only within the same distance class (0, 1, 2, 3+)
dcls = S0.d_full.clip(upper=3).astype(int).to_numpy()
famous = (S0.papers_to2022 > S0.papers_to2022.median()).to_numpy()
o, _ = analyse(S0, famous, dcls, 'Q3 fame: above-median papers vs below, within the same distance class (column rs_fame = adjusted for distance)')
results.append(o)

# ---------------------------------------------------------------- cross-check A: each protein counts once
def per_target(sub, e):
    key = pd.Series(list(zip(sub.tid.to_numpy(), e.astype(int))))
    return (1.0 / key.map(key.value_counts())).to_numpy()
o, _ = analyse(S1, (S1.d_full == 1).to_numpy(), g1, 'Cross-check A: Q1 with each target weighted to count once', weights=per_target)
results.append(o)
o, _ = analyse(S0, (S0.d_full == 0).to_numpy(), g0, 'Cross-check A: Q0 with each target weighted to count once', weights=per_target)
results.append(o)

# ---------------------------------------------------------------- cross-check B: logistic regression
areas = sorted({a.strip() for s in S0.areas for a in s.split(',')})
rows = []
for ph, col in PH:
    k = S0[col].notna()
    sub = S0[k]
    X = pd.DataFrame({f'phase_{ph}': 1.0}, index=sub.index)
    X['d0'] = (sub.d_full == 0).astype(float); X['d1'] = (sub.d_full == 1).astype(float)
    X['fame_sd'] = (sub.lp - S0.lp.mean()) / S0.lp.std()
    for a in areas:
        X[f'area_{a}'] = sub.areas.str.contains(a, regex=False).astype(float)
    X['y'] = sub[col]; X['tid'] = sub.tid
    rows.append(X)
L = pd.concat(rows).fillna(0.0)
ycol = L.pop('y'); grp_t = L.pop('tid')
L = L.loc[:, L.std() > 0]
L = L.drop(columns=[c for c in L if c.startswith('area_') and L[c].sum() < 30])
fit = sm.GLM(ycol, L, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': grp_t})
ci = fit.conf_int()
print(f'\n== Cross-check B: logistic regression, phases I/II/III stacked, n={len(L)}, errors grouped by target')
for v in ('d0', 'd1', 'fame_sd'):
    print(f'  {v:8s} odds ratio {np.exp(fit.params[v]):.3f} (95% CI {np.exp(ci.loc[v, 0]):.3f}-{np.exp(ci.loc[v, 1]):.3f}), p={fit.pvalues[v]:.2g}')
logit_rows = [dict(test='Cross-check B logistic (odds ratio)', phase=v, rs_fame=np.exp(fit.params[v]),
                   rs_fame_l=np.exp(ci.loc[v, 0]), rs_fame_u=np.exp(ci.loc[v, 1]), p_fame_two=fit.pvalues[v]) for v in ('d0', 'd1', 'fame_sd')]
# same model plus literature links as a second fame measure
L2 = L.copy(); lit = np.concatenate([S0.loc[S0[c].notna(), 'lit_links'].to_numpy() for _, c in PH])
L2['litlinks_sd'] = (np.log10(lit + 1) - np.log10(S0.lit_links + 1).mean()) / np.log10(S0.lit_links + 1).std()
fit2 = sm.GLM(ycol, L2, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': grp_t})
ci2 = fit2.conf_int()
print('  + literature links:')
for v in ('d0', 'd1', 'fame_sd', 'litlinks_sd'):
    print(f'  {v:11s} odds ratio {np.exp(fit2.params[v]):.3f} (95% CI {np.exp(ci2.loc[v, 0]):.3f}-{np.exp(ci2.loc[v, 1]):.3f}), p={fit2.pvalues[v]:.2g}')
logit_rows += [dict(test='Cross-check B logistic + literature links (odds ratio)', phase=v, rs_fame=np.exp(fit2.params[v]),
                    rs_fame_l=np.exp(ci2.loc[v, 0]), rs_fame_u=np.exp(ci2.loc[v, 1]), p_fame_two=fit2.pvalues[v]) for v in ('d0', 'd1', 'fame_sd', 'litlinks_sd')]

# ---------------------------------------------------------------- secondary: fame groups by literature links instead of papers
gl = pd.qcut(S1.lit_links, NG, labels=False, duplicates='drop').astype(int).to_numpy()
o, _ = analyse(S1, (S1.d_full == 1).to_numpy(), gl, 'Secondary: Q1 with fame groups by literature links')
results.append(o)

# ---------------------------------------------------------------- secondary: standard network score (Guney z)
zq = (S1.z_full <= S1.z_full.quantile(0.2)).to_numpy()
o, _ = analyse(S1, zq, g1, 'Secondary: standard network score, closest fifth by z vs rest (not DNA-backed)')
results.append(o)

# ---------------------------------------------------------------- calibration of this exact test on fake outcomes
print(f'\n== Calibration of the Q1 test on fake outcomes (1000 sets each; fame odds 1.5 per SD = harsh, 1.17 = strength seen in real data)  ({time.time()-t0:.0f}s)')
z = ((S1.lp - S1.lp.mean()) / S1.lp.std()).to_numpy()
near = (S1.d_full == 1).to_numpy()
def fake(kind):
    G = S1.copy()
    for _, col in PH:
        k = G[col].notna().to_numpy(); base = np.log(G[col].mean() / (1 - G[col].mean()))
        fame_or = {'fame-only': 1.5, 'fame-only (real strength)': 1.17}.get(kind, 1.0)
        lin = base + np.log(fame_or) * z + (np.log(1.3) * near if kind == 'planted' else 0)
        y = (rng.random(len(G)) < 1 / (1 + np.exp(-lin))).astype(float)
        G[col] = np.where(k, y, np.nan)
    return G
cal = []
NCAL = 1000
for kind in ('random', 'fame-only', 'fame-only (real strength)', 'planted'):
    hits_p = hits_f = 0
    for rep in range(NCAL):
        o, _ = analyse(fake(kind), near, g1, kind, draws=2000, boot=2, quiet=True, seed=rep)
        hits_p += o.loc['I-Launch', 'p_plain_higher'] < 0.05; hits_f += o.loc['I-Launch', 'p_fame_higher'] < 0.05
    cal.append(dict(fake=kind, sets=NCAL, plain_shuffle=hits_p / NCAL, fame_aware_20_groups=hits_f / NCAL))
    print(f'  {kind:26s} called significant: plain {hits_p/NCAL:.3f}, fame-aware {hits_f/NCAL:.3f}')

R = pd.concat(results + [pd.DataFrame([q2pair] + logit_rows)], ignore_index=True)
R.to_csv(OUT + 'part2_results.tsv', sep='\t', index=False)
pd.DataFrame(cal).to_csv(OUT + 'part2_calibration.tsv', sep='\t', index=False)
print(f'\ndone {time.time()-t0:.0f}s -> part2_results.tsv, part2_calibration.tsv')
