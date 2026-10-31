# Blind drug-trial predictions (Oct 2026)

## How the test works
- I made the predictions using only information from 2022 or earlier: drug programs as of December 2022, a protein network published in 2021, DNA links found by 2022, and papers per gene counted only up to 2022 (PubMed IDs are given out roughly in date order).
- Every program in phase 1, 2 or 3 in December 2022 gets a predicted chance of moving up. Every target-disease pair that hadn't been tried yet gets a score.
- The files are fingerprinted (SHA-256, in MANIFEST.sha256).
- After posting, I'll open the outcomes (Open Targets releases), record what changed, and score the predictions.

## The predictor
A logistic regression fit only on phase steps that had finished by December 2022. Inputs: whether the target itself is backed by DNA evidence for the disease, whether it is one link away from a DNA-backed gene in the protein network, how close it is to the disease genes compared with proteins that have a similar number of links, and how many papers had been written about it.
P4 (all four inputs) is my primary predictor, chosen on Oct 28, 2026 before any outcome was opened. The main comparison is P4 against P2 (DNA + papers only), which shows whether the network adds anything. The controls are DNA alone (C1), papers alone (C2), the standard network score (C3) and random (C0). All of them are scored the same way.

## Scoring
- Main outcome: programs in phase 1, 2 or 3 in December 2022 that advanced at least one phase or launched by 2026. This shows what ended up working.
- Secondary outcome: new programs started by 2026 that test target-disease pairs that were untried in 2022. This shows where drug developers went next.
- AUC: how often the predictor gives a higher score to a program that moved up than to one that didn't. Computed for the predictor and each control, then compared.
- Relative success: sort programs by predicted score and divide the success rate of the top 10% by the rate of the rest (the same idea as Minikel's 2.6x).
- Calibration: group programs predicted at about x% and check whether about x% of them actually succeeded.
- Resampling: rebuild the dataset 10,000 times by randomly re-picking targets, to check whether the predictor beats each control by more than luck.
- Known weakness: Pharmaprojects and Open Targets are different sources, matched by disease ID and gene, so some reporting differences are expected. I'll check a sample by hand.

## Files
- predictions/active_programs_dec2022.tsv: the 6,560 programs in phase 1-3 in December 2022 and every model's predicted chance of moving up (P4_PRIMARY is the main one).
- predictions/untried_pairs_dec2022.parquet (attached to release v1.0-lock because it's too big for the repo): about 15 million untried protein-disease pairs with their inputs and P4 score.
- predictions/model_coefficients.json: each model's inputs and weights, so any score can be recomputed.
- part2_results/: the history test the predictor was built from.
- code/: the scripts that made everything here.
- MANIFEST.sha256: the fingerprint of every file. The last line is the fingerprint of my frozen research notes (Oct 26), which aren't posted.

## AI Disclosure
The code was written by Claude (Anthropic) at my direction, and Grammarly fixed the grammar for some of my personal notes and write-ups. The research questions, decisions, analyses, planning, and write-ups are mine.

## Data
- Minikel et al. 2024, Nature: drug programs (Citeline Pharmaprojects, pulled Dec 22, 2022) and genetic associations (github.com/ericminikel/genetic_support, CC BY 4.0)
- Gysi et al. 2021, PNAS: human protein network (Barabasi lab)
- NCBI gene2pubmed (downloaded Oct 26, 2026; papers counted up to PMID 36,640,000, the end of 2022)
