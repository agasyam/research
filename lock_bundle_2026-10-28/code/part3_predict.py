"""
Part 3 - build the blind predictions (NOTHING after Dec 2022 is read here).
Training data: every phase step with a known outcome in Minikel's pp.tsv (Pharmaprojects, pulled 22 Dec 2022).
Each candidate model is a logistic regression with phase I / II / III intercepts plus its inputs:
  inputs available (all frozen at 2022):
    dna     target is itself DNA-backed for the disease (distance 0)
    zc      hub-corrected closeness (Guney z, clipped to [-3, 3]) for targets that are NOT DNA-backed; 0 otherwise
    d1      target is exactly 1 link from a DNA-backed gene (raw closeness)
    fame    log10(papers per target up to 2022 + 1), standardised
    zstd    the standard network score: Guney z for every program (clipped to [-5, 5]), no DNA split, no fame
    nodist  no network distance available (target off the map, or disease has no DNA-backed gene)
  candidate predictors (Agasya picks the primary before the lock):
    P1  dna + zc + fame          P2  dna + fame          P3  dna + zc          P4  dna + d1 + zc + fame
  controls (fixed in the plan):  C1 dna alone   C2 fame alone   C3 standard network (zstd)   C0 random
Outputs:
  part3/active_programs_predictions.tsv  every program in phase I/II/III in Dec 2022, P(moves up) per model
  part3/untried_pairs_features.parquet   every network protein x disease with DNA-backed genes that had no
                                          phase I+ program by Dec 2022: inputs + P(success from phase I) per model
  part3/model_coefficients.json          fitted coefficients (so any score can be recomputed)
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
import statsmodels.api as sm

t0 = time.time()
sys.path.insert(0, '/home/claude/atlas/net')
from proximity import degree_bins
from scipy import sparse
from scipy.sparse.csgraph import connected_components

D = '/home/claude/ext/ericminikel_genetic_support/data/'
A = '/home/claude/atlas/'
OUT = A + 'drugs/part3/'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261028)

F = pd.read_csv(A + 'drugs/program_features.tsv', sep='\t')
pp = pd.read_csv(D + 'pp.tsv', sep='\t').dropna(subset=['gene', 'indication_mesh_id']).reset_index(drop=True)
assert (pp.ti_uid.to_numpy() == F.ti_uid.to_numpy()).all()
for c in ['acat', 'ccat', 'succ_1_2', 'succ_2_3', 'succ_3_a', 'indication_mesh_term']:
    F[c] = pp[c]
for c in ['succ_1_2', 'succ_2_3', 'succ_3_a']:
    F[c] = F[c].map({True: 1.0, False: 0.0, 'TRUE': 1.0, 'FALSE': 0.0})

LP_MEAN = np.log10(F.papers_to2022 + 1).mean(); LP_SD = np.log10(F.papers_to2022 + 1).std()

def inputs(d, z, papers):
    d = np.asarray(d, float); z = np.asarray(z, float); papers = np.asarray(papers, float)
    has = np.isfinite(d)
    X = pd.DataFrame({
        'dna': (has & (d == 0)).astype(float),
        'd1': (has & (d == 1)).astype(float),
        'zc': np.where(has & (d >= 1), np.clip(np.nan_to_num(z), -3, 3), 0.0),
        'zstd': np.where(has, np.clip(np.nan_to_num(z), -5, 5), 0.0),
        'fame': (np.log10(papers + 1) - LP_MEAN) / LP_SD,
        'nodist': (~has).astype(float)})
    return X

MODELS = {
    'P1_dna_closeness_fame': ['dna', 'zc', 'fame', 'nodist'],
    'P2_dna_fame':           ['dna', 'fame', 'nodist'],
    'P3_dna_closeness':      ['dna', 'zc', 'nodist'],
    'P4_dna_1link_closeness_fame': ['dna', 'd1', 'zc', 'fame', 'nodist'],
    'C1_dna_alone':          ['dna', 'nodist'],
    'C2_fame_alone':         ['fame'],
    'C3_standard_network':   ['zstd', 'nodist'],
}
PH = [('I', 'succ_1_2'), ('II', 'succ_2_3'), ('III', 'succ_3_a')]

# ---- training table: one row per known phase step (up to Dec 2022)
X0 = inputs(F.d_full, F.z_full, F.papers_to2022)
rows = []
for ph, col in PH:
    k = F[col].notna()
    T = X0[k].copy()
    for p, _ in PH:
        T[f'ph_{p}'] = float(p == ph)
    T['y'] = F.loc[k, col]; T['gene'] = F.loc[k, 'gene']
    rows.append(T)
T = pd.concat(rows, ignore_index=True)
print(f'training steps: {len(T)} (phase I {int(T.ph_I.sum())}, II {int(T.ph_II.sum())}, III {int(T.ph_III.sum())})')
groups = T.gene.astype('category').cat.codes

coefs, fits = {}, {}
for name, cols in MODELS.items():
    Xm = T[['ph_I', 'ph_II', 'ph_III'] + cols]
    fit = sm.GLM(T.y, Xm, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': groups})
    fits[name] = fit
    ci = fit.conf_int()
    coefs[name] = {v: dict(coef=float(fit.params[v]), odds_ratio=float(np.exp(fit.params[v])),
                           l95=float(np.exp(ci.loc[v, 0])), u95=float(np.exp(ci.loc[v, 1])), p=float(fit.pvalues[v]))
                   for v in Xm.columns}
    print(f'{name:30s} ' + '  '.join(f'{v} {np.exp(fit.params[v]):.2f}' for v in cols))
json.dump(dict(models=MODELS, coefficients=coefs, fame_scaling=dict(mean_log10=LP_MEAN, sd_log10=LP_SD),
               training_steps=len(T), note='odds ratios; phase terms are intercepts'),
          open(OUT + 'model_coefficients.json', 'w'), indent=1)

def score(X, phase):
    out = {}
    for name, cols in MODELS.items():
        Xm = X[cols].copy()
        for p, _ in PH:
            Xm.insert(0, f'ph_{p}', float(p == phase) if isinstance(phase, str) else (phase == p).astype(float))
        Xm = Xm[['ph_I', 'ph_II', 'ph_III'] + cols]
        out[name] = fits[name].predict(Xm).to_numpy()
    return out

# ---- primary: programs active in phase I/II/III in Dec 2022
act = F[F.acat.isin(['Phase I', 'Phase II', 'Phase III'])].copy()
ph = act.acat.map({'Phase I': 'I', 'Phase II': 'II', 'Phase III': 'III'}).to_numpy()
Xa = inputs(act.d_full, act.z_full, act.papers_to2022).set_index(act.index)
S = score(Xa, ph)
P = act[['ti_uid', 'gene', 'indication_mesh_id', 'indication_mesh_term', 'acat', 'd_full', 'z_full', 'papers_to2022']].copy()
for k, v in S.items():
    P[k] = np.round(v, 6)
P['C0_random'] = np.round(rng.random(len(P)), 6)
P.to_csv(OUT + 'active_programs_predictions.tsv', sep='\t', index=False)
print(f'active programs scored: {len(P)}  ({time.time()-t0:.0f}s)')
print(P.groupby('acat')[list(S)].mean().round(3).to_string())

# ---- secondary: untried target-disease pairs (no phase I+ program for that exact pair by Dec 2022)
N = A + 'net/out/'
nodes = np.load(N + 'nodes_entrez.npy')
dist = np.load(N + 'dist_full.npy', mmap_mode='r')
deg = np.load(N + 'degree_full.npy')
adj = sparse.load_npz(N + 'adj_full.npz')
_, lab = connected_components(adj, directed=False)
conn = (lab == np.bincount(lab).argmax()) & (deg > 0)
bins = degree_bins(deg)
ann = pd.read_csv('/home/claude/barabasi-lab/covid-19/data/interactome_2019_merged_protAnnots.csv')
e2sym = dict(zip(ann.GeneID, ann.Symbol))
sym2e = dict(zip(ann.Symbol, ann.GeneID))
for gid, syn in zip(ann.GeneID, ann.Synonyms.astype(str)):
    for s in syn.split('|'):
        sym2e.setdefault(s, gid)
nidx = {n: i for i, n in enumerate(nodes)}
fame = pd.read_csv(A + 'fame/gene_fame.tsv', sep='\t').set_index('entrez').papers_to2022
papers_node = np.array([fame.get(e, 0) for e in nodes], float)

# DNA-backed genes per indication: same rule as part1_drugs.py (fixed version)
assoc = pd.read_csv(D + 'assoc.tsv.gz', sep='\t', low_memory=False, usecols=['gene', 'mesh_id', 'source', 'l2g_share', 'year', 'pic_h4'])
keep = (((assoc.source == 'OTG') & (assoc.l2g_share >= 0.5) & assoc.year.between(2005, 2022)) | (assoc.source == 'OMIM')
        | ((assoc.source == 'PICCOLO') & (assoc.pic_h4 >= 0.9)) | (assoc.source == 'Genebass'))
gbt = assoc[keep].groupby('mesh_id').gene.apply(set).to_dict()
sim = pd.read_csv(D + 'sim.tsv.gz', sep='\t'); sim = sim[sim.comb_norm >= 0.8]
similar = sim.groupby('meshcode_a').meshcode_b.apply(set).to_dict()
clinical = set(zip(pp.gene[pp.ccat.isin(['Phase I', 'Phase II', 'Phase III', 'Launched'])],
                   pp.indication_mesh_id[pp.ccat.isin(['Phase I', 'Phase II', 'Phase III', 'Launched'])]))
clin_idx = {}
for g, i in clinical:
    e = sym2e.get(g)
    if e in nidx:
        clin_idx.setdefault(i, set()).add(nidx[e])

inds = sorted(pp.indication_mesh_id.unique())
parts = []
for ii, ind in enumerate(inds):
    gs = set()
    for t in similar.get(ind, set()) | {ind}:
        gs |= gbt.get(t, set())
    gidx = np.array([nidx[sym2e[g]] for g in gs if g in sym2e and sym2e[g] in nidx and conn[nidx[sym2e[g]]]], dtype=int)
    if len(gidx) == 0:
        continue
    m = dist[:, gidx].min(axis=1).astype(float)
    mu = np.zeros(bins.max() + 1); sd = np.zeros(bins.max() + 1)
    for b in range(bins.max() + 1):
        sel = (bins == b) & conn
        if sel.any():
            mu[b] = m[sel].mean(); sd[b] = m[sel].std()
    z = np.where(sd[bins] > 0, (m - mu[bins]) / np.where(sd[bins] > 0, sd[bins], 1), np.nan)
    ok = conn.copy()
    tried = np.zeros(len(nodes), bool)
    if ind in clin_idx:
        tried[list(clin_idx[ind])] = True
    sel = np.where(ok & ~tried)[0]
    parts.append(pd.DataFrame({'node': sel.astype(np.int32), 'ind': np.int16(ii), 'd': m[sel].astype(np.uint8),
                               'z': z[sel].astype(np.float32)}))
U = pd.concat(parts, ignore_index=True)
U['entrez'] = nodes[U.node].astype(np.int64)
U['papers_to2022'] = papers_node[U.node].astype(np.int32)
Xu = inputs(U.d, U.z, U.papers_to2022)
Su = score(Xu, 'I')
for k, v in Su.items():
    U[k] = v.astype(np.float32)
U['indication_mesh_id'] = pd.Categorical.from_codes(U.ind, categories=inds)
U['symbol'] = pd.Categorical([e2sym.get(e, '') for e in U.entrez.to_numpy()])
U = U.drop(columns=['node', 'ind'])
U.to_parquet(OUT + 'untried_pairs_features.parquet', index=False, compression='zstd')
print(f'untried pairs: {len(U):,} across {len(parts)} diseases  ({time.time()-t0:.0f}s)')

# fingerprints
for f in ('active_programs_predictions.tsv', 'untried_pairs_features.parquet', 'model_coefficients.json'):
    h = hashlib.sha256(open(OUT + f, 'rb').read()).hexdigest()
    print(f'SHA-256 {f}: {h}')
