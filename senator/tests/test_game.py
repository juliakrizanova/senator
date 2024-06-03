import numpy as np
from senator.game import Game


def all_go_for_first_region() -> None:
    # There is only one region with non-zero number of votes. No matter the constants, all players should go for it every round.
    votes = [1]
    votes.append(np.zeros((16)))
    game = Game(np.zeros((3, 17)), votes)

    assert game.run_greedy_search()[1].sum() == 0


def total_utility_overflow() -> None:
    # Checking that at the end of the greedy search the total utility does not overflow.
    votes = [100, 100, 100]
    votes.append(np.zeros((14)))
    game = Game(np.zeros((3, 17)), votes)
    assert game.run_greedy_search()[1].sum() <= 300
