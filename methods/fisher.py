import numpy as np
from .common import blocks, unit

def embedding(pool,sl):
    feature=np.asarray(pool.features[sl],dtype=np.float32)
    probability=np.asarray(pool.probability[sl],dtype=np.float32)
    phi=np.column_stack((unit(feature),np.ones(len(feature),np.float32)))
    phi/=np.sqrt(2.)
    return phi*np.sqrt(probability[:,0]*probability[:,1])[:,None]

def select(pool):
    n=len(pool.xyz)
    old=np.unique(np.asarray(pool.initial,dtype=np.int64))
    assert 0<pool.take<=n-len(old)
    used=np.zeros(n,bool);used[old]=True
    info={}
    eligible=np.flatnonzero(~used)
    assert pool.probability.shape == (n, 2) and np.isfinite(pool.probability).all()
    assert np.all(pool.probability >= 0) and np.allclose(pool.probability.sum(1), 1, atol=1e-05)
    d = pool.features.shape[1] + 1
    M = np.eye(d, dtype=np.float64) / d
    if len(old):
        q = embedding(pool, old).astype(np.float64)
        M += q.T @ q
    C = np.zeros((d, d), np.float64)
    for sl in blocks(n):
        assert np.isfinite(pool.features[sl]).all()
        q = embedding(pool, sl).astype(np.float64)
        q = q[~used[sl]]
        C += q.T @ q
    C /= len(eligible)
    inv = np.linalg.inv(M)
    gains = []
    selected = []
    start = float(np.trace(inv @ C))
    for _ in range(pool.take):
        gain = np.empty(n, np.float64)
        for sl in blocks(n):
            q = embedding(pool, sl).astype(np.float64)
            v = q @ inv
            denominator=1 + np.einsum('ij,ij->i', q, v)
            assert np.isfinite(denominator).all() and np.all(denominator>0),'Fisher gain denominator invalid'
            gain[sl] = np.einsum('ij,ij->i', v @ C, v) / denominator
        gain[used] = -np.inf
        i = int(np.argmax(gain))
        assert np.isfinite(gain[i])
        selected.append(i)
        gains.append(float(gain[i]))
        used[i] = True
        q = embedding(pool, np.array([i]))[0].astype(np.float64)
        v = inv @ q
        denominator=float(1 + q @ v)
        assert np.isfinite(denominator) and denominator>0,'Fisher inverse update denominator invalid'
        inv -= np.outer(v, v) / denominator
        assert np.isfinite(inv).all(),'Fisher inverse update has NaN/Inf'
    info.update(regularizer=1 / d, trace_before=start, trace_after=float(np.trace(inv @ C)), trace_reductions=gains, target='mean eligible-voxel Fisher; fixed within batch', approximation='forward greedy normalized last-layer A-optimal, not full BAIT')
    return np.asarray(selected,np.int64),info
