from scipy.stats import binom
import numpy as np
from joblib import delayed,Parallel

def Q(n,p,epsilon):
    k1 = np.floor(n*p+epsilon*n) 
    k2 = np.ceil(n*p-epsilon*n)-1
    return binom.cdf(k1,n,p)-binom.cdf(k2,n,p)

def min_n(p,alpha,epsilon,expand_int=1000):
    n = 1
    q = Q(n,p,epsilon)
    i = 0
    condition = q<1-alpha

    if not condition:
        return n

    while condition:
        r = range(1+i*expand_int,1+(i+1)*expand_int)
        r = np.array(r)
        q_list = np.apply_along_axis(lambda n: Q(n,p,epsilon),0,r)
        condition = all(q_list<1-alpha)
        i += 1
    return r[q_list>=1-alpha].min()

def get_min_ns(alpha,epsilon,p_min=0,p_max=1,n_ps=200):
    ps = np.linspace(p_min,p_max,n_ps)
  
    ns = Parallel(n_jobs=-1)(
        delayed(min_n)(p,alpha,epsilon) for p in ps
    )
  
    return ns,ps

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    start = time.time()
    ns,ps = get_min_ns(0.1,0.05)
    end = time.time()

    print(end-start)

    plt.plot(ps,ns)
    plt.show()
