from senator.utility import *
from senator.game import *
import argparse
import numpy as np


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    parser = argparse.ArgumentParser(
        description="Run the senator game simulation with specified parameters."
    )

    parser.add_argument(
        "--iterations_number",
        type=int,
        default=7,
        help="Number of iterations for which the greedy algorithm is run (default: 7)",
    )

    parser.add_argument(
        "--owner_loss",
        type=float,
        default=0.5,
        help="Loss for the owner (default: 0.5)",
    )
    parser.add_argument(
        "--other_loss", type=float, default=0.0, help="Loss for others (default: 0.0)"
    )
    parser.add_argument(
        "--owner_trashold",
        type=float,
        default=0.35,
        help="Threshold for the owner (default: 0.35). Set to -1 to change the model.",
    )

    args = parser.parse_args()

    votes, initial_utility, is_owner = parse_data(load_data())
    game = Game(
        initial_utility,
        votes,
        is_owner,
        args.num_iterations,
        args.owner_loss,
        args.other_loss,
        args.owner_trashold,
    )
    # game = Game(np.zeros((3,17)),votes)

    actions_in_steps, final_utility, strategies = game.run_greedy_search(
        game.num_iterations
    )
    print(strategies)
    # print(final_utility.sum(axis=1))
    return actions_in_steps, final_utility, strategies


if __name__ == "__main__":
    main()
