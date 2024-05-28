from senator.utility import *
from senator.game import *
import argparse
import numpy as np


def main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    parser = argparse.ArgumentParser(
        description="Run the senator game simulation with specified parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=7,
        help="Number of steps for which the greedy algorithm is run.",
    )

    parser.add_argument(
        "--owner_loss",
        type=float,
        default=0.9,
        help="Loss for the owner.",
    )
    parser.add_argument(
        "--other_loss", type=float, default=0.8, help="Loss for others."
    )
    parser.add_argument(
        "--owner_trashold",
        type=float,
        default=-1,
        help="Threshold for the owner. Set to -1 to change the model.",
    )

    parser.add_argument(
        "--file_path",
        type=float,
        default=FILE_PATH,
        help="Path to the file containing the data.",
    )

    args = parser.parse_args()

    votes, initial_utility, is_owner = parse_data(load_data(args.file_path))
    game = Game(
        initial_utility,
        votes,
        is_owner,
        args.owner_loss,
        args.other_loss,
        args.owner_trashold,
    )
    # game = Game(np.zeros((3,17)),votes)

    actions_in_steps, final_utility, strategies = game.run_greedy_search(args.num_steps)
    return actions_in_steps, final_utility, strategies


if __name__ == "__main__":
    main()
