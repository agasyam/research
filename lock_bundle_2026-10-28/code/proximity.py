"""
Network proximity (Guney et al. 2016, 'closest' measure) for many target sets against one gene set, fast.
  d_c(T, D) = average over targets t in T of the shortest-path distance from t to the nearest gene in D.
  z-score:   compare d_c with random target sets that keep each target's degree (number of interactions):
             each target is swapped for a random protein from the same degree bin (bins of >= 100 proteins,
             as in Guney 2016). Mean and SD of the random d_c are computed exactly from the bin averages
             instead of by 1,000 random draws (same expectation, no sampling noise).
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import numpy as np
from scipy import sparse

def degree_bins(deg, min_bin=100):
    """Group proteins into bins of similar degree, each with >= min_bin proteins (Guney 2016)."""
    order = np.argsort(deg, kind="stable")
    bins = np.empty(len(deg), dtype=int)
    b, count, last = 0, 0, None
    uniq = np.unique(deg[order])
    deg_to_bin = {}
    for d in uniq:
        if count >= min_bin:
            b += 1; count = 0
        deg_to_bin[d] = b
        count += int((deg == d).sum())
    # merge a too-small last bin into the previous one
    if count < min_bin and b > 0:
        deg_to_bin = {d: (bb if bb < b else b - 1) for d, bb in deg_to_bin.items()}
    return np.array([deg_to_bin[d] for d in deg])

def nearest_distance(dist, gene_idx):
    """For every protein: distance to the nearest gene in gene_idx (uint8 matrix, 255 = unreachable)."""
    return dist[:, gene_idx].min(axis=1).astype(float)

def proximity_many(T, m, bins, connected):
    """T: sparse (sets x proteins) 0/1 membership; m: nearest-distance vector; bins: degree bin per protein;
    connected: bool mask of proteins in the largest component (others are ignored, as in Guney 2016).
    Returns d_c, z for every set."""
    T = T.multiply(connected[None, :]).tocsr()
    n = np.asarray(T.sum(1)).ravel()
    mm = np.where(connected, m, 0.0)
    d = (T @ mm) / np.maximum(n, 1)
    nb = bins.max() + 1
    mu_b = np.array([mm[(bins == b) & connected].mean() if ((bins == b) & connected).any() else 0 for b in range(nb)])
    var_b = np.array([mm[(bins == b) & connected].var() if ((bins == b) & connected).any() else 0 for b in range(nb)])
    mu = (T @ mu_b[bins]) / np.maximum(n, 1)
    sd = np.sqrt((T @ var_b[bins])) / np.maximum(n, 1)
    z = (d - mu) / np.where(sd > 0, sd, np.nan)
    d[n == 0] = np.nan; z[n == 0] = np.nan
    return d, z
