import numpy as np

def score(probability):
    p=np.asarray(probability,dtype=np.float32)
    return -(p*np.log(np.maximum(p,1e-12))).sum(-1)

def select(pool, scores):
    assert scores.shape==(len(pool.xyz),) and np.isfinite(scores).all()
    order=np.argsort(-scores,kind='stable')
    chosen=order[~np.isin(order,pool.initial)][:pool.take]
    assert len(chosen)==pool.take
    return chosen,{'selected_score_mean':float(scores[chosen].mean())}
