import numpy as np
from sklearn.cluster import KMeans

def _nearest_unique(centers, features, eligible, take):
    used=set();chosen=[]
    for center in centers:
        best_i=-1;best_d=np.inf
        for start in range(0,len(features),32768):
            block=np.asarray(features[start:start+32768],np.float32)
            dist=np.square(block-center).sum(1)
            if used:
                local=np.asarray([i-start for i in used if start<=i<start+len(block)],int)
                dist[local]=np.inf
            j=int(np.argmin(dist))
            if dist[j]<best_d:best_i=start+j;best_d=float(dist[j])
        if best_i<0:raise RuntimeError('No unused candidate for cluster center')
        used.add(best_i);chosen.append(int(eligible[best_i]))
    if len(chosen)!=take:raise RuntimeError('Could not map all cluster centers to unique candidates')
    return np.asarray(chosen,np.int64)

def clue(features, probability, eligible, take, seed, temperature, n_init, max_iter):
    assert 0<take<=len(eligible),'CLUE take exceeds eligible voxels'
    assert len(np.unique(eligible))==len(eligible),'CLUE eligible duplicate'
    x=np.asarray(features[eligible],np.float32)
    logits=np.log(np.maximum(np.asarray(probability[eligible],np.float64),1e-12))/temperature
    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
    entropy=-(p*np.log(np.maximum(p,1e-12))).sum(1)
    weight=entropy;scale=1.0;uniform_fallback=False
    if weight.max()<1e-6:
        if weight.max()>0:scale=1.0/float(weight.max());weight=weight/weight.max()
        else:uniform_fallback=True;weight=np.ones_like(weight)
    km=KMeans(n_clusters=take,random_state=seed,n_init=n_init,max_iter=max_iter).fit(x,sample_weight=weight)
    chosen=_nearest_unique(km.cluster_centers_,x,eligible,take)
    assert len(chosen)==take and len(np.unique(chosen))==take and np.isin(chosen,eligible).all(),'CLUE cluster mapping invalid'
    return chosen,dict(temperature=temperature,n_init=n_init,max_iter=max_iter,
        weighted_inertia=float(km.inertia_),weight_scale=scale,uniform_weight_fallback=uniform_fallback,candidate_voxels=len(eligible),
        uncertainty_weight_mean=float(entropy.mean()),uncertainty_weight_median=float(np.median(entropy)))


def select(pool, temperature=0.1, n_init=10, max_iter=300):
    eligible=np.setdiff1d(np.arange(len(pool.xyz)),pool.initial)
    return clue(pool.features,pool.probability,eligible,pool.take,pool.seed,temperature,n_init,max_iter)
