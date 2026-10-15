import numpy as np

def permutation(n_voxels, seed):
    return np.random.default_rng(seed).permutation(n_voxels)

def select(order, initial, take):
    old=np.asarray(initial,dtype=np.int64)
    assert len(np.unique(old))==len(old)
    assert np.array_equal(np.sort(order[:len(old)]),np.sort(old))
    picked=np.asarray(order[len(old):len(old)+take],np.int64)
    assert len(picked)==take and not np.isin(picked,old).any()
    return picked,{'sampling':'uniform_unique_voxel_prefix'}
