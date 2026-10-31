"""
Assemble the files Agasya posts publicly to lock the blind predictions (Part 3).
Nothing after Dec 2022 is read. Primary predictor P4 chosen by Agasya on 2026-10-28 00:06 ET.
Written by Claude (AI) for Agasya's STS project; disclose as AI-written code.
"""
import os, json, hashlib, shutil, subprocess
import pandas as pd, numpy as np

A = '/home/claude/atlas/'
L = A + 'lock_2026-10-28/'
shutil.rmtree(L, ignore_errors=True)
for sub in ('predictions', 'code', 'part2_results'):
    os.makedirs(L + sub, exist_ok=True)

# 1. predictions for programs active in Dec 2022 (primary first)
P = pd.read_csv(A + 'drugs/part3/active_programs_predictions.tsv', sep='\t')
P = P.rename(columns={'acat': 'phase_dec2022', 'P4_dna_1link_closeness_fame': 'P4_PRIMARY'})
order = ['ti_uid', 'gene', 'indication_mesh_id', 'indication_mesh_term', 'phase_dec2022', 'd_full', 'z_full', 'papers_to2022',
         'P4_PRIMARY', 'P1_dna_closeness_fame', 'P2_dna_fame', 'P3_dna_closeness',
         'C1_dna_alone', 'C2_fame_alone', 'C3_standard_network', 'C0_random']
P[order].to_csv(L + 'predictions/active_programs_dec2022.tsv', sep='\t', index=False)

# 2. untried pairs: inputs + primary score (other scores follow from the coefficients file)
U = pd.read_parquet(A + 'drugs/part3/untried_pairs_features.parquet')
U = U.rename(columns={'P4_dna_1link_closeness_fame': 'P4_PRIMARY'})
# inputs + primary score only, sorted; every other model's score is recomputable from these inputs and the
# coefficients file (part3_predict.py, function inputs/score), which keeps the file small enough to post
keep = ['entrez', 'indication_mesh_id', 'd', 'z', 'papers_to2022', 'P4_PRIMARY']
U[keep].sort_values(['indication_mesh_id', 'entrez']).to_parquet(L + 'predictions/untried_pairs_dec2022.parquet',
                                                                  index=False, compression='zstd', compression_level=19)

# 3. model definitions + coefficients
C = json.load(open(A + 'drugs/part3/model_coefficients.json'))
C['primary'] = 'P4_dna_1link_closeness_fame'
C['primary_chosen'] = 'by Agasya, 2026-10-28 00:06 ET, after Part 2 and before any 2023-2026 outcome was opened'
C['headline_comparison'] = 'P4 vs P2 (does the network add predictive value beyond DNA evidence and fame?)'
C['input_definitions'] = {
    'dna': 'target is itself DNA-backed for the disease (network distance 0); Minikel 2024 default rule',
    'd1': 'target is exactly 1 link from a DNA-backed gene (Barabasi-lab interactome, Gysi 2021)',
    'zc': 'Guney closest-distance z vs proteins in the same degree bin (>=100 proteins), clipped [-3,3], targets with distance >=1 only',
    'zstd': 'same z for all programs, clipped [-5,5] (standard network method)',
    'fame': '(log10(papers per target up to PMID 36,640,000 + 1) - mean) / SD; gene2pubmed',
    'nodist': 'no network distance (target not on the map, or disease has no DNA-backed gene)'}
json.dump(C, open(L + 'predictions/model_coefficients.json', 'w'), indent=1)

# 4. code that produced the predictions (AI-written, disclosed)
for f in ('net/build_networks.py', 'net/proximity.py', 'fame/build_fame.py', 'drugs/part1_drugs.py',
          'drugs/calibrate_drugs.py', 'drugs/part2_history.py', 'drugs/part3_predict.py', 'make_lock_bundle.py'):
    shutil.copy(A + f, L + 'code/' + os.path.basename(f))
for f in ('part2_results.tsv', 'part2_calibration.tsv', 'part2_output.txt'):
    shutil.copy(A + 'drugs/' + f, L + 'part2_results/' + f)

# 5. fingerprints
plan_pdf = A + 'plan/Agasya_research_notes_2026-10-26.pdf'
plan_hash = hashlib.sha256(open(plan_pdf, 'rb').read()).hexdigest()
rows = []
for root, _, files in os.walk(L):
    for f in sorted(files):
        p = os.path.join(root, f); rel = os.path.relpath(p, L)
        rows.append((hashlib.sha256(open(p, 'rb').read()).hexdigest(), rel, os.path.getsize(p)))
rows.sort(key=lambda r: r[1])
with open(L + 'MANIFEST.sha256', 'w') as fh:
    for h, rel, _ in rows:
        fh.write(f'{h}  {rel}\n')
    fh.write(f'{plan_hash}  (not posted) Agasya_research_notes_2026-10-26.pdf - frozen research plan\n')
for h, rel, sz in rows:
    print(f'{sz/1e6:9.1f} MB  {h[:16]}...  {rel}')
print('plan notes', plan_hash)
print('rows: active', len(P), 'untried', len(U))
