from scipy.stats import binom
import numpy as np
from joblib import delayed,Parallel
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm

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

def sample_size_prop(alpha,epsilon):
    min_n_ = lambda p: -min_n(p,alpha,epsilon)
    bounds = Bounds([0],[1])
    res = minimize(min_n_,0.5,method='trust-constr',bounds=bounds)
    return -res.fun

def sample_size_prop_normal(alpha,epsilon,p=0.5):

    z = norm.ppf(1-alpha/2)
    n = ((z/epsilon)**2)*p*(1-p)
    n = int(np.ceil(n))
    return n

def get_min_ns(alpha,epsilon,p_min=0,p_max=1,n_ps=200):
    ps = np.linspace(p_min,p_max,n_ps)
  
    ns = Parallel(n_jobs=-1)(
        delayed(min_n)(p,alpha,epsilon) for p in ps
    )
  
    return ns,ps

def simulated_gurantee(n,epsilon,n_sim=1000):
  p_true = 0.5
  x = np.random.binomial(n=n,p=p_true,size=int(n_sim))
  p_hat = x/n
  return (np.abs(p_hat-p_true)<=epsilon).mean()

if __name__ == '__main__':
    import pandas as pd

    alpha_list = [0.15,0.1,0.05,0.02,0.01]
    epsilon_list = [0.15,0.1,0.05,0.02,0.01]

    df = pd.DataFrame()
    for a,e in zip(alpha_list,epsilon_list):
        direct = sample_size_prop(alpha=a,epsilon=e)
        normal = sample_size_prop_normal(alpha=a,epsilon=e)
        direct_guarantee = simulated_gurantee(n=direct,epsilon=e,n_sim=1e+6)
        normal_guarantee = simulated_gurantee(n=normal,epsilon=e,n_sim=1e+6)
        df = pd.concat([
            df, 
            pd.DataFrame({
                'alpha':[a],
                'epsilon':[e], 
                'direct':[direct], 
                'normal':[normal], 
                'direct_guarantee':[direct_guarantee],
                'normal_guarantee':[normal_guarantee],
            })
        ])
    
    print(df)
    

    
