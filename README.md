# Active learning selectors for TLS leaf–wood separation

Acquisition-strategy code for *Comparing active learning strategies for leaf–wood separation in terrestrial laser scanning point clouds: label budget, class-specific errors, and run-to-run variability* (Yeonho Kim and Choong-Sik Kim).

This package contains only the selection code: Random, Entropy, BALD, k-Center, BADGE, CLUE, and Fisher A-optimal. The PTv3 training and evaluation pipeline, data, model weights, and logs are not included, so the paper's trained-model results cannot be reproduced from this package alone. The acquisition algorithms follow the experiment code. Comments, docstrings, unused `Pool` fields, and unused local variables were removed. `clue.py` also has a numerical guard for the case where all candidates are nearly certain.

## Run

```bash
pip install -r requirements.txt
python example.py
```

Checked with Python 3.12.14, NumPy 2.2.6, and scikit-learn 1.7.2. `requirements.txt` does not pin versions; scikit-learn KMeans results may differ between versions.

## Usage

Each call selects labels for one training tree. Build `methods.common.Pool(xyz, features, mass, initial, take, seed, probability=probability)` (pass `probability` by keyword):

- `features`: `(N, 64)` decoder features; `probability`: `(N, 2)` class probabilities (0 = leaf, 1 = wood).
- `initial`: indices of already labeled candidates; `take`: number to select; `mass`: unused (pass ones).
- Output is `(indices, metadata)`; indices are candidate rows, not raw-point IDs.

| Method | Call |
| --- | --- |
| Entropy | `entropy.select(pool, entropy.score(probability))` |
| BALD | `bald.select(pool, bald.score(mc_probability))`, `mc_probability` is `(T, N, 2)` from T = 5 passes with the model in eval mode and only DropPath modules in train mode |
| k-Center, BADGE, CLUE, Fisher | `select(pool)` |
| Random | `order = random.permutation(N, seed)`, then `random.select(order, initial, take)` (`initial` must be the first `len(initial)` entries of `order`) |

## Notes

- Candidate: the original point nearest to the mean coordinate (centroid) of each occupied 0.015 m voxel. Candidate construction and model inference are outside this package.
- CLUE: temperature 0.1, `n_init=10`, `max_iter=300` (scikit-learn KMeans; versions may change results). If all candidates are nearly certain, the entropy weights are rescaled and the factor is returned as `weight_scale` (`weighted_inertia` then refers to the rescaled weights).
- Fisher A-optimal: L2-normalized last-layer features plus a bias coordinate (divided by √2), weighted by √(p₀p₁), regularization 1/65 (D = 64), forward greedy selection. It approximates BAIT and does not use backward pruning.
- Training in the study was not deterministic.
