import numpy as np
import pandas as pd
from senator.game import Game
from senator.utility import load_data, parse_data


def compute_order(utility_sum: np.ndarray, player: int) -> int:
    """
    Computes the order of a given player based on sum of utilities over all resources.
    """
    order = 0

    # print(f"u_s:{utility_sum}")
    for i in range(3):
        if utility_sum[i - 1] > utility_sum[player]:
            order += 1
    # print(f"order:{order}")
    return order


def search_loss_parameters(
    owner_loss_lower: float = 0.6,
    owner_loss_upper: float = 0.9,
    other_loss_lower: float = 0.6,
    other_loss_upper: float = 0.9,
    owner_increment: int = 0.1,  # length of increment step
    other_increment: int = 0.1,  # length of increment step
    num_steps: int = 7,
    file_path: str = "experiment_results.csv",
) -> None:
    """
    Does a search over loss parameters and saves outcomes of the model to a csv file.
    Specifically, intervals for loss parameters are given together with incremental steps for each interval.
    Then the final utility after num_steps steps is computed for each combination of loss parameters.
    """
    votes, initial_utility, is_owner = parse_data(load_data())

    results = []
    mixed_strategies = 0

    # number of iterations for cycles over loss parameters
    num_owner_iterations = int(
        np.floor((owner_loss_upper - owner_loss_lower) / owner_increment)
    )
    num_other_iterations = int(
        np.floor((other_loss_upper - other_loss_lower) / other_increment)
    )

    # cycle accross parameters
    for i in range(num_owner_iterations + 1):  # TODO: no for?
        num_win = 0
        owner_loss = owner_loss_lower + (
            i * ((owner_loss_upper - owner_loss_lower) / num_owner_iterations)
        )
        for j in range(num_other_iterations + 1):

            other_loss = other_loss_lower + (
                j * ((other_loss_upper - other_loss_lower) / num_other_iterations)
            )
            print(f"iteration:{(i,j)}")
            print(f"owner_loss:{owner_loss}")
            print(f"other_loss:{other_loss}")
            game = Game(
                initial_utility,
                votes,
                is_owner,
                num_steps,
                owner_loss,
                other_loss,
                2,
            )
            _, utility, strategies_in_steps = game.run_greedy_search()

            utility_sum = np.floor(utility.sum(axis=1))

            results.append([i] + [owner_loss, other_loss] + utility_sum.tolist())

            if compute_order(utility_sum, 0) < 2:
                num_win += 1
            print(f"round: {(i*num_owner_iterations+1) + (j+1)}")
            print(f"win_ratio: {num_win / (j+1)}")

    # save results to a csv file
    columns = ["iteration", "owner_loss", "other_loss", "u1", "u2", "u3"]

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(file_path, index=False)
