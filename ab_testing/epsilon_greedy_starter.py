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
        for i, m in self.machines:
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
        while self.N_trials <= max_trials:
            self.run_trial()
            if self.N_trials % update_every == 0:
                self.progress_update() 
        self.plot_experiment_results()
        self.reset_experiment()


    def reset_experiment(self):
        self.N_trials = 0 
        for machine in self.machines:
            machine.N_plays = 0 
            machinee.win_rate = None








if __name__ == "__main__":

    probas = [0.2, 0.5, 0.8]
    epsilon = 0.05
    