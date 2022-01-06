import csv 
import numpy as np 


from collections import defaultdict
from scipy.stats import norm, t 
from scipy.stats.mstats import ttest_ind
from statsmodels.stats.weightstats import ztest


if __name__ == "__main__":

    DATA_FILE = "advertisement_clicks.csv"
    with open(DATA_FILE, newline = "") as csvfile:
        reader = csv.DictReader(csvfile)
        data_dict = defaultdict(list)
        for row in reader:
            group = row["advertisement_id"]
            outcome = int(row["action"])
            data_dict[group].append(outcome)

    XA = np.array(data_dict["A"], dtype=int)
    XB = np.array(data_dict["B"], dtype=int)

    print("Z-test by hand like a GANGSTAAAAAAA")
    x_bar = (XA - XB).mean()
    sig2_A = XA.var(ddof = 1)
    sig2_B = XB.var(ddof = 1)
    sig_bar = np.sqrt(sig2_A / len(XA) + sig2_B / len(XB))
    z = x_bar / sig_bar
    tail_prob = 1 - norm.cdf(abs(z), loc = 0, scale = 1)
    p_val =  2 * tail_prob 
    print(f"The Z score is {z:0.8} with a p-value of {p_val:0.8%}")


    print("\nComparing with statsmodels....")
    z_sm, p_sm = ztest(XA, XB)
    print(f"The Z score is {z_sm:0.8} with a p-value of {p_sm:0.8%}")

    print(f"CTR for ad A: {XA.mean():0.8%}")
    print(f"CTR for ad B: {XB.mean():0.8%}")


    if p_val < 0.05 and abs(p_val - p_sm) < 1e-5:
        print("\nYo son, this difference is hella significant, dawg")


    print("\nWould a T-Test be different?")
    print(f"Sample variance for ad A: {sig2_A:0.8%}")
    print(f"Sample variance for ad B: {sig2_B:0.8%}")
    print("Assuming equal variances")
    t_val = x_bar / sig_bar
    df_numer = (sig2_A / len(XA) + sig2_B / len(XB))**2
    df_denom = (sig2_A / len(XA))**2 / (len(XA) - 1) + (sig2_B / len(XB))**2 / (len(XB) - 1)
    deg_free = df_numer / df_denom
    tail_prob = 1 - t.cdf(abs(t_val), df = deg_free)
    p_val =  2 * tail_prob 
    print(f"The t-score is {t_val:0.8} with a p-value of {p_val:0.8%}")

    print("\nComparing with scipy....")
    z_sp, p_sp = ttest_ind(a = XA, b = XB, equal_var = True)
    print(f"The t-score is {z_sp:0.8} with a p-value of {p_sp:0.8%}")

    if p_val < 0.05 and abs(p_val - p_sp) < 1e-5:
        print("\nYo son, this difference is hella significant, dawg")








