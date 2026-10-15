import numpy as np

def score(mc_probability):
    p=np.asarray(mc_probability,dtype=np.float32)
    mean=p.mean(0)
    h=-(mean*np.log(np.maximum(mean,1e-12))).sum(-1)
    conditional=-(p*np.log(np.maximum(p,1e-12))).sum(-1).mean(0)
    return np.maximum(h-conditional,0)

def select(pool, scores):
    assert scores.shape==(len(pool.xyz),) and np.isfinite(scores).all()
    order=np.argsort(-scores,kind='stable')
    chosen=order[~np.isin(order,pool.initial)][:pool.take]
    assert len(chosen)==pool.take
    return chosen,{'selected_score_mean':float(scores[chosen].mean())}
