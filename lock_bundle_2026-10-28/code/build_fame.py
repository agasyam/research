"""
Fame per gene from NCBI gene2pubmed (downloaded by Agasya, Oct 26 2026).
papers_all     = number of PubMed papers linked to the gene
papers_to2022  = same, counting only PMIDs <= 36,640,000 (PMID 36,600,000 was first published 2022-12-23 and
                 36,650,000 on 2023-01-01, per Europe PMC), i.e. papers available by the end of 2022.
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT = 36_640_000
pm = pickle.load(open('human_gene2pubmed.pkl', 'rb'))
rows = [(g, len(v), sum(1 for p in v if p <= CUT)) for g, v in pm.items()]
F = pd.DataFrame(rows, columns=['entrez', 'papers_all', 'papers_to2022'])
F.to_csv('gene_fame.tsv', sep='\t', index=False)
N = '/home/claude/atlas/net/out/'
nodes = np.load(N + 'nodes_entrez.npy')
f = F.set_index('entrez').reindex(nodes).fillna(0)
print('network proteins with >=1 paper:', int((f.papers_all > 0).sum()), 'of', len(nodes))
print('papers per protein: median', int(f.papers_to2022.median()), '| top 1% above', int(f.papers_to2022.quantile(0.99)))
for net in ('full', 'literature', 'systematic'):
    d = np.load(N + f'degree_{net}.npy')
    m = d > 0
    r = spearmanr(d[m], f.papers_to2022.to_numpy()[m]).statistic
    print(f'{net:11s} links vs papers (proteins in that half): Spearman {r:.2f}  (n={m.sum()})')
