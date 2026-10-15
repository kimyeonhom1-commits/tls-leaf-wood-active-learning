import numpy as np
from .common import blocks

def embedding(pool,sl):
    f=np.asarray(pool.features[sl],dtype=np.float32)
    probability=np.asarray(pool.probability[sl],dtype=np.float32)
    residual=probability.copy()
    residual[np.arange(len(probability)),probability.argmax(1)]-=1
    gradient=(residual[:,:,None]*f[:,None,:]).reshape(len(f),-1)
    assert gradient.shape==(len(f),2*f.shape[1]),'BADGE gradient dimension mismatch'
    assert np.isfinite(gradient).all(),'BADGE gradient has NaN/Inf'
    return gradient

def update_distance(pool,distance,index):
    anchor=embedding(pool,np.array([index]))[0]
    for sl in blocks(len(distance)):
        d=np.square(embedding(pool,sl)-anchor).sum(1)
        distance[sl]=np.minimum(distance[sl],d)

def select(pool):
    n=len(pool.xyz)
    old=np.unique(np.asarray(pool.initial,dtype=np.int64))
    assert 0<pool.take<=n-len(old)
    used=np.zeros(n,bool);used[old]=True
    picked=[];info={}
    norm = np.empty(n, np.float64)
    for sl in blocks(n):
        norm[sl] = np.square(embedding(pool, sl)).sum(1)
    norm[used] = -np.inf
    distance = np.full(n, np.inf)
    rng = np.random.default_rng(pool.seed)
    i = int(np.argmax(norm))
    for t in range(pool.take):
        if t:
            distance[used] = 0
            total = distance.sum()
            i = int(rng.choice(n, p=distance / total)) if total > 0 else int(np.flatnonzero(~used)[0])
        picked.append(i)
        used[i] = True
        update_distance(pool, distance, i)
    return np.asarray(picked,np.int64),info
