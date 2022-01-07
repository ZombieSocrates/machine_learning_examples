import pathlib 
import numpy as np   

from matplotlib import pyplot as plt
from typing import List

class SlotMachine(object):


    def __init__(self, true_win_rate: float):
        self.true_win_rate = true_win_rate
        self.N_plays = 0 
        self.win_rate = None  


    def pull_arm(self):
        pull_val = np.random.uniform(size = 1)[0]
        return 1 if pull_val <= self.true_win_rate else 0 
  

    def update_win_rate(self, value):
        self.N_plays += 1 
        prior = 0 if self.win_rate is None else self.win_rate
        updated =  1 / self.N_plays * (value + (self.N_plays - 1) * prior)
        self.win_rate = updated


    def play(self):
        pull_val = self.pull_arm()
        self.update_win_rate(value = pull_val)


    def print_summary(self, show_true_rate = False):
        print(f"\tTimes played: {self.N_plays}")
        if self.win_rate is not None:
            print(f"\tCurrent Win Rate: {self.win_rate:0.4%}")
        if show_true_rate:
            print(f"\tTrue Win Rate: {self.true_win_rate:0.2%}")


class EpsilonGreedyExperiment(object):


    def __init__(self, machine_win_rates: List[float], eps: float):
        self.eps = eps 
        self.machines = [SlotMachine(true_win_rate = p) for p in machine_win_rates]
        self.N_trials = 0


    def choose_random_machine(self):
        return np.random.choice(self.machines)


    def choose_best_machine(self):
        win_sort = lambda x: x.win_rate if x.win_rate is not None else -1
        sorted_machines = sorted(self.machines, key = win_sort)
        return sorted_machines[-1]


    def progress_update(self):
        for i, m in enumerate(self.machines):
            print(f"Machine {i + 1}")
            m.print_summary(show_true_rate = False)
            print("------"*5, "\n")


    def run_trial(self):
        eps_draw = np.random.uniform(size = 1)[0]
        if self.N_trials == 0 or eps_draw <= self.eps:
            machine = self.choose_random_machine()
        else:
            machine = self.choose_best_machine()
        machine.play()
        self.N_trials += 1


    def run_experiment(self, max_trials: float, update_every: int):
        while self.N_trials < max_trials:
            self.run_trial()
            if self.N_trials % update_every == 0:
                self.progress_update()
        print("Experiment Complete! Saving results and clearing data.") 
        self.plot_experiment_results()
        self.reset_experiment()


    def plot_experiment_results(self, lbl_pad = 10):
        plays_per_machine = [x.N_plays for x in self.machines]
        win_rates = [x.win_rate for x in self.machines]
        coords = [i + 1 for i in range(len(win_rates))]
        fig, ax = plt.subplots(figsize = (10, 10))
        plt.bar(x = coords, height = plays_per_machine)
        ax.set_xticks(coords)
        ax.set_xticklabels([f"Machine {i}" for i in coords], rotation = 65,
            ha = "right")
        ax.set_ylabel("Number of Plays")
        for idx, wr in enumerate(win_rates):
            ax.text(x = coords[idx], y = plays_per_machine[idx] + lbl_pad, 
                s = f"{wr:0.4%}", ha = "center")
        total_trials = sum(plays_per_machine)
        plt.title(f"Epsilon Greedy: {total_trials} Trials; {self.eps:0.2%} Base Epsilon")
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path())
        print(save_path)
        plt.savefig(save_path)


    def get_experiment_save_path(self, base_dir = "viz/epsilon_greedy", ext = ".png"):
        base_pth = pathlib.Path(base_dir)
        if not base_pth.exists():
            base_pth.mkdir(exist_ok = True, parents = True)
        exp_index = len([k for k in base_pth.glob(f"*{ext}")])
        next_index = str(exp_index + 1).rjust(2, "0")
        exp_file = base_pth / f"experiment_{next_index}_results{ext}"
        return exp_file


    def reset_experiment(self):
        self.N_trials = 0 
        for machine in self.machines:
            machine.N_plays = 0 
            machine.win_rate = None


if __name__ == "__main__":

    probas = [0.2, 0.5, 0.8]
    epsilon = 0.05
    max_trials = 10000
    epg_lab = EpsilonGreedyExperiment(machine_win_rates = probas, eps = epsilon)
    epg_lab.run_experiment(max_trials = max_trials, update_every = 1000)

