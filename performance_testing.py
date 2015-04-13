# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

def score(real_adj,est_adj):
    n=np.shape(real_adj)[0]
    d=real_adj-est_adj
    d=np.abs(d)
    res=np.sum(d)
    res=res/(n*(n-1))
    return res

