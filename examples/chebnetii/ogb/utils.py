import math
import random
import numpy as np
import pgl
import paddle

def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2
