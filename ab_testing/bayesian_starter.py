import pathlib 
import numpy as np   

from matplotlib import pyplot as plt
from scipy.stats import beta
from typing import List


class SlotMachine(object):

    '''lol this is more or less the same as the eps greedy slot

    TODO: allow this to support both real and binary valued rewards

    '''
    def __init__(self, true_win_rate: float):
        self.true_win_rate = true_win_rate
        self.N_plays = 0 
        self.alpha = 1 
        self.beta = 1
        self.win_rate = None 


    def make_posterior_pdf(self):
        q = np.linspace(0.01, 0.99, 100)
        p = [beta.pdf(d, a = self.alpha, b = self.beta) for d in q]
        return q, p 


    def pull_arm(self):
        pull_val = np.random.uniform(size = 1)[0]
        return 1 if pull_val <= self.true_win_rate else 0 
  

    def update_win_rate(self, value):
        self.N_plays += 1
        self.alpha += value 
        self.beta += 1 - value
        prior = 0 if self.win_rate is None else self.win_rate
        updated =  1 / self.N_plays * (value + (self.N_plays - 1) * prior)
        self.win_rate = updated


    def play(self):
        pull_val = self.pull_arm()
        self.update_win_rate(value = pull_val)


    def print_summary(self, show_true_rate = False, plays_to_exclude:int = 0):
        print(f"\tTimes played: {self.N_plays - plays_to_exclude}")
        if self.win_rate is not None:
            print(f"\tCurrent Win Rate: {self.win_rate:0.4%}")
        if show_true_rate:
            print(f"\tTrue Win Rate: {self.true_win_rate:0.2%}")
        posterior = f"Beta(a = {self.alpha}, b = {self.beta})"
        print(f"\tCurrent Posterior Distribution: {posterior}")



class ThompsonSamplingExperiment(object):

    def __init__(self, machine_win_rates: List[float]):
        self.machines = [SlotMachine(true_win_rate = p) for p in machine_win_rates]
        self.N_trials = 0

        
    def get_thompson_sample(self, machine):
        tsamp = beta.rvs(a = machine.alpha, b = machine.beta, size =1)[0]
        return tsamp


    def choose_best_machine(self):
        thompson_sort = lambda x: self.get_thompson_sample(x)
        sorted_machines = sorted(self.machines, key = thompson_sort)
        return sorted_machines[-1]


    def progress_update(self):
        for i, m in enumerate(self.machines):
            print(f"Machine {i + 1}")
            m.print_summary(show_true_rate = False)
            print("------"*5, "\n")


    def run_trial(self):
        if self.N_trials < len(self.machines):
            machine = self.machines[self.N_trials]
        else:
            machine = self.choose_best_machine()
        machine.play()
        self.N_trials += 1


    def count_optimal_plays(self):
        true_win_rate_sort = lambda x: x.true_win_rate
        sorted_machines = sorted(self.machines, key = true_win_rate_sort)
        best_machine = sorted_machines[-1]
        return best_machine.N_plays


    def count_total_wins(self):
        wins = 0
        for m in self.machines:
            w = m.win_rate * m.N_plays
            wins += w 
        return int(wins)


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
        print(f"Number of winning plays: {self.count_total_wins()}")
        self.reset_experiment()


    def plot_experiment_progress(self, n_trials, lbl_pad:int = 10):
        fig, ax = plt.subplots(figsize = (10, 10))
        for m in self.machines:
            X, y = m.make_posterior_pdf()
            true_wr = f"real p: {m.true_win_rate:0.0%}" 
            obs_wr = f"win rate: {m.alpha - 1}/{m.N_plays}"
            m_lbl = f"{true_wr}; {obs_wr}"
            plt.plot(X, y, label = m_lbl)
        ax.set_ylabel("Density")
        the_title = f"Thompson Sampling: Posterior Distributions at {n_trials} Trials"
        plt.title(the_title)
        plt.legend(loc = "best")
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path(n_trials = n_trials))
        print(f"Saving details to {save_path}")
        plt.savefig(save_path)


    def get_experiment_save_path(self, n_trials, base_dir = "viz/bayes_bandit", 
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
            machine.N_plays = 0
            machine.win_rate = None
            machine.alpha = 1
            machine.beta = 1

 


if __name__ == "__main__":

    probas = [0.05, 0.33, 0.20, 0.10]
    max_trials = 10000
    tsampling_lab = ThompsonSamplingExperiment(machine_win_rates = probas)
    tsampling_lab.run_experiment(max_trials = max_trials, update_every = 1000)   