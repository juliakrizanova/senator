import numpy as np
import experiments.parameter_search as ps
from senator.utility import *
from senator.game import *
from experiments.plotting import *


def main():
    print(f"portion of wins:{ps.search_loss_parameters(0.6,0.9,0.6,0.9,0.01,0.01,7,"experiment_results4.csv")}")
    #plot_experiment_results("experiment_results4.csv")


if __name__ == "__main__":
    main()
