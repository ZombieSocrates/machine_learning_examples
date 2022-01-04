import pdb
import numpy as np 
from scipy.stats import norm, t


def conf_interval_experiment(N, mu, sd, conf_level, use_tdist = True):
    '''Takes a sample, constructs a confidence interval for the mean using that
    sample, returns True if the interval catches the true population mean, or
    False otherwise 
    '''
    X = norm.rvs(loc = mu, scale = sd, size = N)
    x_bar = X.mean()
    sd_hat = X.std(ddof = 1)
    critical_val = conf_level + (1 - conf_level)/2
    if use_tdist:
        coef = t.ppf(critical_val, df = N - 1)
    else:
        coef = norm.ppf(critical_val, loc = 0, scale = 1)
    lower = x_bar - coef * sd_hat / np.sqrt(N)
    upper = x_bar + coef * sd_hat / np.sqrt(N)
    return (mu <= upper) & (mu >= lower)



if __name__ == "__main__":

    np.random.seed(1)


    N = 1000
    mu = 5
    sd = 2
    conf_level = 0.95

    X = norm.rvs(loc = mu, scale = sd, size = N)

    print("Z-confidence interval")
    x_bar = X.mean()
    sd_hat = X.std(ddof = 1)
    critical_val = conf_level + (1 - conf_level)/2
    z_coef = norm.ppf(critical_val, loc = 0, scale = 1)
    lower = x_bar - z_coef * sd_hat / np.sqrt(N)
    upper = x_bar + z_coef * sd_hat / np.sqrt(N)

    print(f"{conf_level:%} confidence interval: [{lower:0.8f}, {upper:0.8f}]")
    print(f"mu hat is {x_bar:0.8f}")


    print("t-confidence interval")
    t_coef = t.ppf(critical_val, df = N - 1)
    lower = x_bar - t_coef * sd_hat / np.sqrt(N)
    upper = x_bar + t_coef * sd_hat / np.sqrt(N)
    print(f"{conf_level:%} confidence interval: [{lower:0.8f}, {upper:0.8f}]")


    print("Building 100 confidence intervals")
    experiment_results = []
    for i in range(100):
        trial = conf_interval_experiment(N = N, mu = mu, sd = sd, 
            conf_level = conf_level, use_tdist = True)
        experiment_results.append(trial)

    print(f"Confidence interval captured mu {sum(experiment_results)} times")
    pdb.set_trace()