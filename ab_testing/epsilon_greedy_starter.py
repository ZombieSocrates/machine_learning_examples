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
        self.eps_init = eps 
        self.machines = [SlotMachine(true_win_rate = p) for p in machine_win_rates]
        self.N_trials = 0
        self.N_explores = 0
        self.N_exploits = 0


    def choose_random_machine(self):
        return np.random.choice(self.machines)


    def choose_best_machine(self):
        win_sort = lambda x: x.win_rate if x.win_rate is not None else -1
        sorted_machines = sorted(self.machines, key = win_sort)
        return sorted_machines[-1]


    def decay_epsilon(self, decay_type: List[str]):
        '''
        Arithmetic implies eps = max(eps_init - k*N, min_eps)
        Exponential implies eps = eps_init * alpha ^ N
        '''
        if decay_type not in ["Arithmetic", "Exponential"]:
            raise NotImplementedError
        elif decay_type == "Arithmetic":
            min_eps = 1e-6
            scaling = 1e-5
            decayed = self.eps_init - scaling * self.N_trials
            self.eps = max(decayed, min_eps)
        else:
            alpha = 0.9999
            self.eps = self.eps_init * alpha**self.N_trials


    def progress_update(self):
        for i, m in enumerate(self.machines):
            print(f"Machine {i + 1}")
            m.print_summary(show_true_rate = False)
            print("------"*5, "\n")


    def run_trial(self, eps_decay_type:str = None):
        eps_draw = np.random.uniform(size = 1)[0]
        if self.N_trials == 0 or eps_draw <= self.eps:
            machine = self.choose_random_machine()
            self.N_explores += 1
        else:
            machine = self.choose_best_machine()
            self.N_exploits += 1
        machine.play()
        self.N_trials += 1
        if eps_decay_type is not None:
            self.decay_epsilon(decay_type = eps_decay_type)


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


    def run_experiment(self, max_trials: float, update_every: int, 
        eps_decay_type:str = None):
        while self.N_trials < max_trials:
            self.run_trial(eps_decay_type = eps_decay_type)
            if self.N_trials % update_every == 0:
                print(f"PROGRESS AT {self.N_trials} trials")
                self.progress_update()
                if eps_decay_type is not None:
                    print(f"Current epsilon: {self.eps:0.10}")
                print("*****"*10,"\n")
        print("Experiment Complete! Saving results and clearing data.")
        print(f"Number of times explored: {self.N_explores}")
        print(f"Number of times exploited: {self.N_exploits}")
        print(f"Number of optimal plays: {self.count_optimal_plays()}")
        print(f"Number of winning plays: {self.count_total_wins()}")
        if eps_decay_type is not None:
            print(f"Final value of epsilon: {self.eps:0.10}")
        self.plot_experiment_results(eps_decay_type = eps_decay_type)
        self.reset_experiment()


    def plot_experiment_results(self, eps_decay_type:str=None, lbl_pad:int = 10):
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
        the_title = f"Epsilon Greedy: {total_trials} Trials; {self.eps_init:0.2%} Base Epsilon"
        if eps_decay_type is not None:
            the_title = f"{the_title}; {eps_decay_type} Epsilon Decay"
        plt.title(the_title)
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path())
        print(f"Saving details to {save_path}")
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
        self.N_explores = 0
        self.N_exploits = 0 
        for machine in self.machines:
            machine.N_plays = 0 
            machine.win_rate = None


if __name__ == "__main__":

    probas = [0.2, 0.4, 0.8, 0.6]
    epsilon = 0.10
    max_trials = 10000
    epg_lab = EpsilonGreedyExperiment(machine_win_rates = probas, eps = epsilon)
    epg_lab.run_experiment(max_trials = max_trials, update_every = 1000,
        eps_decay_type = "Exponential")

