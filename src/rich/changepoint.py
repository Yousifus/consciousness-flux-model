# src/rich/changepoint.py
import numpy as np

def segment_changes(x, max_k=3, min_len=5):
    """Binary segmentation by SSE minimization on x (1D)."""
    x = np.asarray(x, float)
    n = len(x)
    cps = []

    def sse(a, b):
        seg = x[a:b]
        mu = seg.mean()
        return ((seg - mu)**2).sum()

    def split(a, b, k_left):
        if k_left == 0 or (b - a) < 2*min_len:
            return
        best_i, best_gain = None, 0.0
        full = sse(a,b)
        for i in range(a+min_len, b-min_len):
            gain = full - (sse(a,i)+sse(i,b))
            if gain > best_gain:
                best_gain = gain; best_i = i
        if best_i is not None:
            cps.append(best_i)
            split(a, best_i, k_left-1)
            split(best_i, b, k_left-1)

    split(0, n, max_k)
    cps.sort()
    return cps
