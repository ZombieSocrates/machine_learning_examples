import pathlib 
import numpy as np   

from matplotlib import pyplot as plt
from scipy.stats import dirichlet
from typing import List


class DiceToRoll(object):

    def __init__(self, true_probas: List[float], dirichlet_init:int = 1):
        '''Initialize by default with alpha as a vector of ones
        '''
        if not self.validate_probas(true_probas):
            raise ValueError("Invalid Probability Array")
        self.true_probas = true_probas
        self.die_faces = [i + 1 for i in range(len(self.true_probas))]
        self.N_rolls = 0
        self.d_init = dirichlet_init
        self.alphas = [self.d_init for i in self.true_probas]

        
    def validate_probas(self, event_probs: List[float]):
        ep = np.array(event_probs)
        valid_range = np.all((ep >= 0) & (ep <= 1))
        valid_sum = np.sum(ep) == 1
        return valid_sum & valid_range


    def value_from_roll(self):
        return np.random.choice(a = self.die_faces, p = self.true_probas) 


    def update(self, value):
        '''Pretty sure the dirichlet update just adds the counts of all the 
        times you've seen a particular face land
        '''
        self.N_rolls += 1
        index_to_update = value - 1
        self.alphas[index_to_update] += 1


    def roll(self):
        roll_val = self.value_from_roll()
        self.update(value = roll_val)


    def print_summary(self, show_true_dist = True):
        print(f"\tTimes Rolled: {self.N_rolls}")
        if show_true_dist:
            print("\ttruly distributed as")
            for i, k in enumerate(self.die_faces):
                print(f"\t\t{k}: {self.true_probas[i]::.2%}")
        dist_str = f"Dirichlet(alpha = {','.join([str(k) for k in self.alphas])}"
        print(f"\tCurrent Posterior Distribution:({dist_str})")
        print("\tempirical counts")
        for i, k in enumerate(self.die_faces):
            print(f"\t\t{k} has been rolled {self.alphas[i] - self.d_init} times")


    def get_posterior_mean(self):
        '''plot this out at various time steps to see how it compares to 
        self.true_probas'''
        return dirichlet.mean(alpha = self.alphas).tolist()



class DiceExperiment(object):

    def __init__(self, dice_probs: List[float]):
        self.die = DiceToRoll(true_probas = dice_probs)
        self.N_trials = 0


    def run_trial(self):
        self.die.roll()
        self.N_trials += 1


    def run_experiment(self, max_trials: float, update_every: int):
        while self.N_trials < max_trials:
            self.run_trial()
            if self.N_trials % update_every == 0:
                print(f"PROGRESS AT {self.N_trials} trials")
                self.die.print_summary(show_true_dist = False)
                print("*****"*10,"\n")
                self.plot_experiment_progress(n_trials = self.N_trials)
        print("Experiment Complete! Saving results and clearing data.")
        self.reset_experiment()


    def plot_experiment_progress(self, n_trials, lbl_pad:int = 10, width:float = 0.3):
        '''https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
        '''
        fig, ax = plt.subplots(figsize = (10, 10))
        
        xlabels = [str(k) for k in self.die.die_faces]
        true_ps = self.die.true_probas
        post_ps = self.die.get_posterior_mean()
        x = np.arange(len(xlabels))

        bar_true = ax.bar(x - width/2, true_ps, width, label = "True Categorical Distribution")
        bar_post = ax.bar(x + width/2, post_ps, width, label = "Learned Posterior Distribution")
        ax.bar_label(bar_true, padding = lbl_pad, fmt = "%.2f")
        ax.bar_label(bar_post, padding = lbl_pad, fmt = "%.2f")
        ax.set_xticks(x, xlabels)
        ax.set_ylabel("Probability Mass")
        ttl1 = f"Rolling a {len(self.die.die_faces)}-sided Die"
        ttl2 = f"True Vs Posterior Distributions at {n_trials} Trials"
        plt.title("\n".join([ttl1, ttl2]))
        plt.legend(loc = "best")
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path(n_trials = n_trials))
        print(f"Saving details to {save_path}")
        plt.savefig(save_path)


    def get_experiment_save_path(self, n_trials, base_dir = "viz/categorical_dice", 
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
        self.die.N_rolls = 0
        n_faces = len(self.die.die_faces)
        self.die.alphas = [self.die.d_init] * n_faces



if __name__ == "__main__":

    dice_weights = [0.2, 0.3, 0.05, 0.25, 0.1, 0.1]
    max_trials = 1000
    dice_lab = DiceExperiment(dice_probs = dice_weights) 
    dice_lab.run_experiment(max_trials = max_trials, update_every = 100)  



