"""
Build three versions of the human protein interaction network from the Barabasi-lab merged interactome
(Gysi et al. 2021 PNAS, Dataset S2: 18,505 proteins, 327,924 interactions, each tagged with its source databases),
and the all-pairs shortest-path distance matrix for each.
  full        every interaction
  systematic  interactions found by at least one proteome-scale screen that tested pairs without choosing
              them from the literature (HI-union/HuRI yeast two-hybrid, BioPlex and QUBIC affinity purification,
              CoFrac co-fractionation)
  literature  interactions with no systematic-screen support (found only in literature-curated databases)
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path, connected_components

SRC = Path("/home/claude/barabasi-lab/covid-19/data/DatasetS2.csv")
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
SYSTEMATIC = {"hiunion", "bioplex", "qubic", "CoFrac"}

e = pd.read_csv(SRC)
e = e[e.proteinA_entrezid != e.proteinB_entrezid]
srcs = e.databases.astype(str).str.split("|")
e["systematic"] = srcs.apply(lambda s: bool(SYSTEMATIC & set(s)))
nodes = np.array(sorted(set(e.proteinA_entrezid) | set(e.proteinB_entrezid)))
nid = {n: i for i, n in enumerate(nodes)}
np.save(OUT / "nodes_entrez.npy", nodes)

def adj(df):
    a = df.proteinA_entrezid.map(nid).to_numpy(); b = df.proteinB_entrezid.map(nid).to_numpy()
    M = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes))).tocsr()
    M = ((M + M.T) > 0).astype(np.int8)
    return M

versions = {"full": e, "systematic": e[e.systematic], "literature": e[~e.systematic]}
rows = []
for name, df in versions.items():
    M = adj(df)
    deg = np.asarray(M.sum(1)).ravel()
    ncomp, lab = connected_components(M, directed=False)
    lcc = np.bincount(lab[deg > 0]).max() if (deg > 0).any() else 0
    sparse.save_npz(OUT / f"adj_{name}.npz", M.tocsr())
    np.save(OUT / f"degree_{name}.npy", deg)
    rows.append(dict(network=name, proteins_with_edges=int((deg > 0).sum()), interactions=int(M.nnz // 2), largest_component=int(lcc)))
    if "--dist" in sys.argv:
        t0 = time.time()
        D = np.full((len(nodes), len(nodes)), 255, dtype=np.uint8)
        for s in range(0, len(nodes), 512):
            idx = np.arange(s, min(s + 512, len(nodes)))
            d = shortest_path(M, method="D", unweighted=True, directed=False, indices=idx)
            d[np.isinf(d)] = 255
            D[idx] = np.minimum(d, 255).astype(np.uint8)
        np.save(OUT / f"dist_{name}.npy", D)
        print(name, "distance matrix", f"{time.time()-t0:.0f}s", flush=True)
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(OUT / "network_versions.tsv", sep="\t", index=False)
