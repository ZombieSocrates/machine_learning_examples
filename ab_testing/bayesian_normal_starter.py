import pathlib 
import numpy as np   

from matplotlib import pyplot as plt
from scipy.stats import norm
from typing import List


class SlotMachineGaussian(object):

    '''Assuming that each of these begins with a standard normal distribution. 
    Expressing things in terms of precision, lambda.

    Precision equal 1 / variance, remember.
    '''
    def __init__(self, true_mean: float, true_precision: float):
        self.mu_true = true_mean
        if true_precision <= 0:
            msg = f"Precision must be positive, not {true_precision}"
            raise ValueError(msg)
        self.prec_true = true_precision
        self.N_plays = 0
        self.rewards = 0 
        self.mu = 0
        self.prec = 1


    def variance_true(self):
        return 1/ self.prec_true


    def variance_posterior(self):
        return 1/ self.prec


    def average_reward(self):
        if self.N_plays == 0:
            return 0 
        return self.rewards / self.N_plays


    def make_posterior_pdf(self):
        ppfs = np.linspace(0.01, 0.99, 100)
        sd =np.sqrt(self.variance_posterior())
        Xs = [norm.ppf(p, loc = self.mu, scale = sd) for p in ppfs]
        Ys = [norm.pdf(x, loc = self.mu, scale = sd) for x in Xs]
        return Xs, Ys


    def pull_arm(self):
        sd_true =  np.sqrt(1 / self.prec_true)
        return norm.rvs(loc = self.mu_true, scale = sd_true, size = 1)[0]
  

    def update_posterior(self, value, known_tau:float = 1.0):
        '''I think these updates are correct but I could sure be wrong
        '''
        self.N_plays += 1
        self.rewards += value
        prior_prec = self.prec
        self.prec = known_tau + prior_prec
        self.mu = 1 / self.prec * (known_tau * value + self.mu * prior_prec)


    def play(self, known_tau:float = 1.0):
        pull_val = self.pull_arm()
        self.update_posterior(value = pull_val, known_tau = known_tau)


    def print_summary(self, show_true_dist = False):
        print(f"\tTimes Played: {self.N_plays}")
        print(f"\tExpected Reward: {self.average_reward():.2f}")
        if show_true_dist:
            params_true = f"{self.mu_true:.2f}, {self.variance_true():.2f}"
            print(f"\tTruly Distributed as N({params_true})")
        params_obs = f"{self.mu:.2f}, {self.variance_posterior():.2f}"
        print(f"\tCurrent Posterior Distribution: N({params_obs})")



class GaussianThompsonSamplingExperiment(object):

    def __init__(self, machine_params: List[tuple], known_tau: float = 1):
        self.machines = self.set_up_machines(machine_params)
        self.N_trials = 0
        if known_tau <= 0:
            msg = f"Constant precision must be positive, not {known_tau}"
            raise ValueError(msg)
        self.tau = known_tau

    
    def set_up_machines(self, param_pairs: List[tuple]):
        slots = []
        for mu, prec in param_pairs:
            slots.append(SlotMachineGaussian(mu, prec))
        return slots
        

    def get_thompson_sample(self, machine):
        machine_sd = np.sqrt(machine.variance_posterior())
        tsamp = norm.rvs(loc = machine.mu, scale = machine_sd, size =1)[0]
        return tsamp


    def choose_best_machine(self):
        thompson_sort = lambda x: self.get_thompson_sample(x)
        sorted_machines = sorted(self.machines, key = thompson_sort)
        return sorted_machines[-1]


    def progress_update(self):
        for i, m in enumerate(self.machines):
            print(f"Machine {i + 1}")
            m.print_summary(show_true_dist = True)
            print("------"*5, "\n")


    def run_trial(self):
        if self.N_trials < len(self.machines):
            machine = self.machines[self.N_trials]
        else:
            machine = self.choose_best_machine()
        machine.play(known_tau = self.tau)
        self.N_trials += 1


    def count_optimal_plays(self):
        true_mu_sort = lambda x: x.mu_true
        sorted_machines = sorted(self.machines, key = true_mu_sort)
        best_machine = sorted_machines[-1]
        return best_machine.N_plays


    def count_total_payout(self):
        payouts = [m.rewards for m in self.machines]
        return sum(payouts)


    def run_experiment(self, max_trials: float, update_every: int):
        while self.N_trials < max_trials:
            self.run_trial()
            if self.N_trials % update_every == 0:
                print(f"PROGRESS AT {self.N_trials} trials")
                self.progress_update()
                print("*****"*10,"\n")
                self.plot_experiment_progress(n_trials = self.N_trials)
        print("Experiment Complete! Saving results and clearing data.")
        print(f"Number of optimal plays: {self.count_optimal_plays()}")
        print(f"Total Payout: {self.count_total_payout():.2f}")
        self.reset_experiment()


    def plot_experiment_progress(self, n_trials, lbl_pad:int = 10):
        fig, ax = plt.subplots(figsize = (10, 10))
        for m in self.machines:
            X, y = m.make_posterior_pdf()
            true_var = m.variance_true()
            true_params = f"true dist N({m.mu_true}, {true_var})" 
            avg_payout = f"average payout {m.average_reward():.4f}"
            m_lbl = f"{true_params}; {avg_payout}"
            plt.plot(X, y, label = m_lbl)
        ax.set_ylabel("Density")
        ttl1 = f"Gaussian Thompson Sampling; Tau assumed {self.tau}"
        ttl2 = f"Posterior Distributions at {n_trials} Trials"
        plt.title("\n".join([ttl1, ttl2]))
        plt.legend(loc = "best")
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path(n_trials = n_trials))
        print(f"Saving details to {save_path}")
        plt.savefig(save_path)


    def get_experiment_save_path(self, n_trials, base_dir = "viz/bayes_bandit_gaussian", 
        ext = ".png"):
        base_pth = pathlib.Path(base_dir)
        if not base_pth.exists():
            base_pth.mkdir(exist_ok = True, parents = True)
        # gross
        curr_pngs = [str(k) for k in base_pth.glob(f"*{ext}")]
        unique_exp = set([p.split("-")[0] for p in curr_pngs])
        count_experiments = len(unique_exp) 
        if len(curr_pngs) % 10 == 0:
            count_experiments += 1 
        next_index = str(count_experiments).rjust(2, "0")
        exp_file = base_pth / f"experiment_{next_index}-{n_trials}trials{ext}"
        return exp_file


    def reset_experiment(self):
        self.N_trials = 0
        for machine in self.machines:
            self.N_plays = 0
            self.rewards = 0 
            self.mu = 0
            self.prec = 1

 

if __name__ == "__main__":

    tau_val = 1
    param_pairs = [(1, 1), (2, 1), (3, 1)]
    max_trials = 1000
    tsampling_lab = GaussianThompsonSamplingExperiment(machine_params = param_pairs,
        known_tau = tau_val)
    tsampling_lab.run_experiment(max_trials = max_trials, update_every = 100)   