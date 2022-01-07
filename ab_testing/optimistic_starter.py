import pathlib 
import numpy as np   

from matplotlib import pyplot as plt
from typing import List


class SlotMachine(object):

    '''lol this is more or less the same as the eps greedy slot

    TODO: allow this to support both real and binary valued rewards

    '''
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


    def print_summary(self, show_true_rate = False, plays_to_exclude:int = 0):
        print(f"\tTimes played: {self.N_plays - plays_to_exclude}")
        if self.win_rate is not None:
            print(f"\tCurrent Win Rate: {self.win_rate:0.4%}")
        if show_true_rate:
            print(f"\tTrue Win Rate: {self.true_win_rate:0.2%}")


class OptimisticInitialExperiment(object):

    def __init__(self, machine_win_rates: List[float], optim_default:float = 2.0):
        self.machines = [SlotMachine(true_win_rate = p) for p in machine_win_rates]
        self.optim_default = optim_default
        self.set_optimistic_initial_vals()
        self.N_trials = 0
        # self.N_explores = 0
        # self.N_exploits = 0


    def set_optimistic_initial_vals(self, optim_val:float = None):
        '''Not only does the initial value get set to 1, but we also
        count this initial value as a first trial. If you don't do this, 
        it's possible to lose on a machine and then never return to it
        [see experiment results 1 - 3]
        '''
        val = self.optim_default if optim_val is None else optim_val
        for m in self.machines:
            m.win_rate = val
            m.N_plays = 1



    def choose_random_machine(self):
        '''in this case, we would only do this on the very first trial when
        every optimistic win rate is equal'''
        return np.random.choice(self.machines)


    def choose_best_machine(self):
        win_sort = lambda x: x.win_rate if x.win_rate is not None else -1
        sorted_machines = sorted(self.machines, key = win_sort)
        return sorted_machines[-1]


    def progress_update(self):
        for i, m in enumerate(self.machines):
            print(f"Machine {i + 1}")
            m.print_summary(show_true_rate = False, plays_to_exclude = 1)
            print("------"*5, "\n")


    def run_trial(self):
        if self.N_trials == 0:
            machine = self.choose_random_machine()
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
            w = m.win_rate * (m.N_plays - 1)
            wins += w 
        return int(wins)


    def run_experiment(self, max_trials: float, update_every: int, 
        optim_val:float = None):
        '''you aren't really exploring with this the same way you were with
        epsilon-greedy, or at least, to the extent that you are, it's all 
        encapsulated in the initial value
        '''
        if optim_val is not None:
            self.set_optimistic_initial_vals(optim_val)
        while self.N_trials < max_trials:
            self.run_trial()
            if self.N_trials % update_every == 0:
                print(f"PROGRESS AT {self.N_trials} trials")
                self.progress_update()
                print("*****"*10,"\n")
        print("Experiment Complete! Saving results and clearing data.")
        # print(f"Number of times explored: {self.N_explores}")
        # print(f"Number of times exploited: {self.N_exploits}")
        print(f"Number of optimal plays: {self.count_optimal_plays()}")
        print(f"Number of winning plays: {self.count_total_wins()}")
        val_used = self.optim_default if optim_val is None else optim_val
        self.plot_experiment_results(initial_value = val_used)
        self.reset_experiment()


    def plot_experiment_results(self, initial_value, lbl_pad:int = 10):
        # remember, every experiment initializes each machine with one play
        plays_per_machine = [x.N_plays - 1 for x in self.machines]
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
        the_title = f"Optimistic Initial Values: {total_trials} Trials; Initial Value: {initial_value}"
        plt.title(the_title)
        plt.tight_layout()
        save_path = str(self.get_experiment_save_path())
        print(f"Saving details to {save_path}")
        plt.savefig(save_path)


    def get_experiment_save_path(self, base_dir = "viz/optimistic_init", ext = ".png"):
        base_pth = pathlib.Path(base_dir)
        if not base_pth.exists():
            base_pth.mkdir(exist_ok = True, parents = True)
        exp_index = len([k for k in base_pth.glob(f"*{ext}")])
        next_index = str(exp_index + 1).rjust(2, "0")
        exp_file = base_pth / f"experiment_{next_index}_results{ext}"
        return exp_file


    def reset_experiment(self):
        self.N_trials = 0
        self.set_optimistic_initial_vals()

 


if __name__ == "__main__":

    probas = [0.5, 0.3, 0.55, 0.72, 0.19]
    max_trials = 10000
    op_init_lab = OptimisticInitialExperiment(machine_win_rates = probas)
    op_init_lab.run_experiment(max_trials = max_trials, update_every = 1000,
        optim_val = 10)   