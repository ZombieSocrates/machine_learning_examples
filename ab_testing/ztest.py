import numpy as np 
from scipy.stats import norm 
from statsmodels.stats.weightstats import ztest



if __name__ == "__main__":

    np.random.seed(0)

    N = 100 
    mu = 0.2 
    sd = 1

    X = norm.rvs(loc = mu, scale = sd, size = N)

    print("Two-sided Z Test")
    mu_hat = X.mean()
    sig_hat = X.std(ddof = 1)
    z = mu_hat / (sig_hat / np.sqrt(N))
    tail_prob = 1 - norm.cdf(abs(z), loc = 0, scale = 1) #Can also do norm.sf(value) to get upper tail prob
    p_value = 2 * tail_prob
    print(f"The Z score is {z:0.8f}, yielding a p-value of {p_value:0.8%}") 

    sm_z, sm_p = ztest(X)
    print(f"Statsmodels gives {sm_z:0.8f}, {sm_p:0.8%}")


    print("\nOne-sided Z Test")
    print(f"The one-sided p-value is simply {tail_prob:0.8%}") 
    
    sm_z, sm_p = ztest(X, alternative = "larger")
    print(f"Statsmodels gives {sm_z:0.8f}, {sm_p:0.8%}")

    
    print(f"\nNull reference value of {mu}")
    z = (mu_hat - mu) / (sig_hat / np.sqrt(N))
    tail_prob = norm.sf(abs(z), loc = 0, scale = 1)
    p_value = 2 * tail_prob
    print(f"The Z score is {z:0.8f}, yielding a p-value of {p_value:0.8%}")
    sm_z, sm_p = ztest(X, value = mu)
    print(f"Statsmodels gives {sm_z:0.8f}, {sm_p:0.8%}")


    print(f"\nTwo sample test")

    mu1 = 0.2
    sig1 = 1
    X1 = norm.rvs(loc = mu1, scale = sig1, size = N)

    mu2 = 0.5
    sig2 = 1 
    X2 = norm.rvs(loc = mu2, scale = sig2, size = N)

    x_bar = (X1 - X2).mean()
    sig_diff = np.sqrt(X1.var(ddof = 1) / N + X2.var(ddof = 1)/N)
    z = x_bar / (sig_diff)
    tail_prob = norm.sf(abs(z), loc = 0, scale = 1)
    p_value = 2 * tail_prob
    print(f"The Z score is {z:0.8f}, yielding a p-value of {p_value:0.8%}")
    
    sm_z, sm_p = ztest(X1, X2)
    print(f"Statsmodels gives {sm_z:0.8f}, {sm_p:0.8%}")








    import pdb 
    pdb.set_trace()