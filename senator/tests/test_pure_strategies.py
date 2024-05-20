import numpy as np
from senator.game import Game
from senator.utility import load_data, parse_data, FILE_PATH


def test_pure_strategy() -> None:
    initial_utility = np.random.randint(0, 100, size=(3, 17))
    votes = np.random.randint(300, 1000, size=17)
    is_owner = np.random.randint(0, 1, size=(3, 17))

    game_fixed = Game(initial_utility, votes, is_owner, 0.9, 0.5, -1)
    game_trashold = Game(initial_utility, votes, is_owner, 0.9, 0.5, 0.35)

    strategies_fixed = game_fixed.run_greedy_search()[2]
    strategies_trashold = game_trashold.run_greedy_search()[2]

    assert np.all(
        np.isin(strategies_fixed, [0, 1])
    ), "The fixed-ownership model should have pure strategies"
    assert np.all(
        np.isin(strategies_trashold, [0, 1])
    ), "The trashold-ownership model should have pure strategies"


def test_pure_strategy_real_game() -> None:
    votes, initial_utility, is_owner = parse_data(load_data(FILE_PATH))
    iterations = 1000
    for _ in range(iterations):
        owner_loss = np.random.randint(1, 101) * 0.01
        other_loss = np.random.randint(1, 101) * 0.01
        owner_trashold = np.random.randint(1, 101) * 0.01
        game = Game(
            initial_utility, votes, is_owner, owner_loss, other_loss, owner_trashold
        )
        strategies = game.run_greedy_search()[2]
        assert np.all(
            np.isin(strategies, [0, 1])
        ), f"The senator game does not have pure strategies for: own:{owner_loss}, oth:{other_loss}, trh:{owner_trashold} strategies: {strategies}"
