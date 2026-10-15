import numpy as np
from methods.common import Pool
from methods import random, entropy, bald, kcenter, badge, clue, fisher


def main():
    rng = np.random.default_rng(42)
    n, dim, take = 128, 64, 4
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    mc = rng.dirichlet([2, 2], size=(5, n)).astype(np.float32)
    probability = mc.mean(axis=0)
    order = random.permutation(n, seed=42)
    initial = order[:2]
    pool = Pool(xyz=xyz, features=features, mass=np.ones(n),
                initial=initial, take=take, seed=42, probability=probability)
    results = {
        "Random": random.select(order, initial, take),
        "Entropy": entropy.select(pool, entropy.score(probability)),
        "BALD": bald.select(pool, bald.score(mc)),
        "k-Center": kcenter.select(pool),
        "BADGE": badge.select(pool),
        "CLUE": clue.select(pool),
        "Fisher A-optimal": fisher.select(pool),
    }
    for name, (indices, _) in results.items():
        assert indices.shape == (take,)
        assert len(np.unique(indices)) == take
        assert not np.isin(indices, initial).any()
        assert ((indices >= 0) & (indices < n)).all()
        print(f"{name}: {indices.tolist()}")


if __name__ == "__main__":
    main()
