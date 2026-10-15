import numpy as np
from .common import blocks, unit

def update_distance(pool,distance,index):
    anchor=unit(pool.features[np.array([index])])[0]
    for sl in blocks(len(distance)):
        d=np.square(unit(pool.features[sl])-anchor).sum(1)
        distance[sl]=np.minimum(distance[sl],d)

def select(pool):
    n=len(pool.xyz)
    old=np.unique(np.asarray(pool.initial,dtype=np.int64))
    assert 0<pool.take<=n-len(old)
    used=np.zeros(n,bool);used[old]=True
    picked=[];info={}
    distance = np.full(n, np.inf)
    for i in old:
        update_distance(pool, distance, i)
    if not len(old):
        raise ValueError('feature_kcenter requires the shared initial labels')
    for _ in range(pool.take):
        distance[used] = -np.inf
        i = int(np.argmax(distance))
        picked.append(i)
        used[i] = True
        update_distance(pool, distance, i)
    return np.asarray(picked,np.int64),info
