from dataclasses import dataclass
import numpy as np
CHUNK=32768

@dataclass
class Pool:
    xyz: np.ndarray
    features: np.ndarray
    mass: np.ndarray
    initial: np.ndarray
    take: int
    seed: int
    probability: np.ndarray = None

def blocks(n):
    for b in range(0, n, CHUNK):
        yield slice(b, min(b + CHUNK, n))

def unit(f):
    f = np.array(f, dtype=np.float32, copy=True)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

