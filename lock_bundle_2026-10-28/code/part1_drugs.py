"""
Part 1 for the drug programs: features only, NO trial outcomes are read here.
  - DNA-backed genes per disease, Minikel 2024 default rule (pipeline_best defaults): MeSH similarity >= 0.8;
    sources OMIM + GWAS (OTG, PICCOLO, Genebass), somatic intOGen excluded; OTG only if L2G share >= 0.5 and
    year 2005-2022; PICCOLO only if H4 >= 0.9.
    FIX 2026-10-27: v1 (Oct 26) applied the year filter to every source, which silently dropped all OMIM rows
    (OMIM has no year), and did not apply the PICCOLO H4 filter. Found when Part 2's replication gave 1.85x
    instead of Minikel's ~2.6x. v1 kept as part1_drugs_v1_oct26.py.
  - For every program (target x disease): distance from the target to the nearest DNA-backed gene, and a
    z-score against proteins in the same degree bin, on the full network and on each half.
  - Fame of each target: papers up to 2022 (gene2pubmed), literature-only links, programs per target.
Then a calibration on FAKE outcomes: does the plain shuffle or the fame-group shuffle get fooled when
"successes" follow fame only? And can they catch a planted closeness signal?
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import sys, time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, '/home/claude/atlas/net')
from proximity import degree_bins

t0 = time.time()
G = '/home/claude/ext/ericminikel_genetic_support/data/'
N = '/home/claude/atlas/net/out/'
OUT = '/home/claude/atlas/drugs/'

nodes = np.load(N + 'nodes_entrez.npy'); nidx = {n: i for i, n in enumerate(nodes)}
ann = pd.read_csv('/home/claude/barabasi-lab/covid-19/data/interactome_2019_merged_protAnnots.csv')
sym2e = dict(zip(ann.Symbol, ann.GeneID))
for gid, syn in zip(ann.GeneID, ann.Synonyms.astype(str)):
    for s in syn.split('|'):
        sym2e.setdefault(s, gid)
def idx(sym):
    e = sym2e.get(sym); return nidx.get(e) if e is not None else None

# ---- programs (outcome columns deliberately NOT loaded)
pp = pd.read_csv(G + 'pp.tsv', sep='\t', usecols=['ti_uid', 'gene', 'indication_mesh_id'])
pp = pp.dropna(subset=['gene', 'indication_mesh_id'])
pp['j'] = pp.gene.map(idx)

# ---- DNA-backed genes per indication (Minikel default rule, frozen at 2022)
assoc = pd.read_csv(G + 'assoc.tsv.gz', sep='\t', low_memory=False, usecols=['gene', 'mesh_id', 'source', 'l2g_share', 'year', 'pic_h4'])
keep = (((assoc.source == 'OTG') & (assoc.l2g_share >= 0.5) & assoc.year.between(2005, 2022))
        | (assoc.source == 'OMIM')
        | ((assoc.source == 'PICCOLO') & (assoc.pic_h4 >= 0.9))
        | (assoc.source == 'Genebass'))
print('associations kept by source:', assoc[keep].source.value_counts().to_dict())
assoc = assoc[keep]
genes_by_trait = assoc.groupby('mesh_id').gene.apply(set).to_dict()
sim = pd.read_csv(G + 'sim.tsv.gz', sep='\t'); sim = sim[sim.comb_norm >= 0.8]
similar = sim.groupby('meshcode_a').meshcode_b.apply(set).to_dict()
dna = {}
for ind in pp.indication_mesh_id.unique():
    gs = set()
    for t in similar.get(ind, set()) | {ind}:
        gs |= genes_by_trait.get(t, set())
    dna[ind] = gs
print(f'indications {len(dna)}, with >=1 DNA-backed gene: {sum(1 for v in dna.values() if v)}  ({time.time()-t0:.0f}s)')

# ---- distances and z-scores on each network
feat = pp[['ti_uid', 'gene', 'indication_mesh_id']].copy()
for net in ('full', 'systematic', 'literature'):
    dist = np.load(N + f'dist_{net}.npy', mmap_mode='r')
    deg = np.load(N + f'degree_{net}.npy')
    adj = sparse.load_npz(N + f'adj_{net}.npz')
    _, lab = connected_components(adj, directed=False)
    big = np.bincount(lab).argmax(); conn = (lab == big) & (deg > 0)
    bins = degree_bins(deg)
    d_col = np.full(len(pp), np.nan); z_col = np.full(len(pp), np.nan)
    for ind, grp in pp.groupby('indication_mesh_id'):
        gidx = np.array([j for j in (idx(g) for g in dna[ind]) if j is not None and conn[j]], dtype=int)
        if len(gidx) == 0:
            continue
        m = dist[:, gidx].min(axis=1).astype(float)
        mu = np.zeros(bins.max() + 1); sd = np.zeros(bins.max() + 1)
        for b in range(bins.max() + 1):
            sel = (bins == b) & conn
            if sel.any():
                mu[b] = m[sel].mean(); sd[b] = m[sel].std()
        for row, j in zip(grp.index, grp.j):
            if pd.notna(j) and conn[int(j)]:
                j = int(j); pos = pp.index.get_loc(row)
                d_col[pos] = m[j]
                z_col[pos] = (m[j] - mu[bins[j]]) / sd[bins[j]] if sd[bins[j]] > 0 else np.nan
    feat[f'd_{net}'] = d_col; feat[f'z_{net}'] = z_col
    print(f'{net}: programs with a distance {np.isfinite(d_col).sum()}  ({time.time()-t0:.0f}s)')

# ---- fame
fame = pd.read_csv('/home/claude/atlas/fame/gene_fame.tsv', sep='\t').set_index('entrez')
feat['entrez'] = feat.gene.map(sym2e)
feat['papers_to2022'] = feat.entrez.map(fame.papers_to2022).fillna(0)
lit_deg = np.load(N + 'degree_literature.npy')
feat['lit_links'] = [lit_deg[int(j)] if pd.notna(j) else np.nan for j in pp.j]
feat['programs_per_target'] = feat.groupby('gene').ti_uid.transform('size')
feat['fame_group'] = pd.qcut(feat.papers_to2022.rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
feat['dna_genes_for_disease'] = feat.indication_mesh_id.map(lambda i: len(dna[i]))
feat.to_csv(OUT + 'program_features.tsv', sep='\t', index=False)

# descriptive (no outcomes): are famous targets closer?
f = feat[np.isfinite(feat.d_full)]
print('\nShare of programs whose target is 0 / 1 / 2+ links from a DNA-backed gene (full network), by fame group:')
tab = pd.crosstab(f.fame_group, f.d_full.clip(upper=2), normalize='index').round(3)
tab.columns = ['d=0', 'd=1', 'd>=2']; print(tab.to_string())
print(f'done {time.time()-t0:.0f}s')
