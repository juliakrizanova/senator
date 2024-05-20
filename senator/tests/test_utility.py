import numpy as np
from senator.utility import (
    get_utility,
    get_utility_matrix,
    load_data,
    parse_data,
    FILE_PATH,
)


def test_data_shape() -> None:
    assert len(load_data(FILE_PATH).shape) == 2


def test_get_utility_shape() -> None:
    # Checking that the utility has the correct shape
    assert get_utility(
        np.zeros((3, 17)), [1, 2, 3], np.zeros((17)), np.zeros((17))
    ).shape == (3, 17)


def test_get_utility_computation() -> None:
    # Checking that the utility is correctly computed
    votes = np.zeros((17))
    votes[:3] = [100, 100, 100]
    utility_sum = get_utility(np.zeros((3, 17)), [0, 1, 2], votes).sum(axis=1)
    assert np.array_equal(utility_sum, [100, 100, 100])


def test_get_utility_owner_loss() -> None:
    # Checking the utility is correctly computed based on the owner_loss in trashold-ownership model
    previous_utility = np.zeros((3, 17))
    previous_utility[0] = np.ones(17)
    assert (
        get_utility(
            previous_utility, [1, 1, 1], np.ones(17), np.zeros((3, 17)), 0.5, 0, 1
        )[0][0]
        == 0.5
    )

    # Checking the utility is correctly computed based on the owner_loss in fixed-ownership model
    previous_utility = np.zeros((3, 17))
    previous_utility[0] = np.ones(17)
    is_owner = np.zeros((3, 17))
    is_owner[0] = np.ones(17)
    assert (
        get_utility(previous_utility, [1, 1, 1], np.ones(17), is_owner, 1, 0, 2)[0][0]
        == 1
    )


def test_get_utility_other_loss() -> None:
    # Checking the utility is correctly computed based on the other_loss in trashold-ownership model
    previous_utility = np.zeros((3, 17))
    previous_utility[0] = 0.5 * np.ones(17)
    assert (
        get_utility(
            previous_utility, [1, 1, 1], np.ones(17), np.zeros((3, 17)), 0, 1, 1
        )[0][0]
        == 0.5
    )

    # Checking the utility is correctly computed based on the other_loss in fixed-ownership model
    previous_utility = np.zeros((3, 17))
    previous_utility[0] = np.ones(17)
    assert (
        get_utility(
            previous_utility, [1, 1, 1], np.ones(17), np.zeros((3, 17)), 0, 1, 2
        )[0][0]
        == 1
    )


def test_get_utility_matrix_shape() -> None:
    # Checking that the utility matrix has the correct shape
    assert get_utility_matrix(
        np.zeros((3, 17)), np.zeros((17)), np.zeros((3, 17))
    ).shape == (3, 17, 17, 17)


def test_get_utility_matrix() -> None:
    # Checking that the utility matrix is correctly computed
    votes = np.zeros((17))
    votes[:3] = [300, 300, 300]
    utility_matrix = get_utility_matrix(np.zeros((3, 17)), votes, np.zeros((3, 17)))
    utility_vector = utility_matrix[:, 0, 0, 0]
    # All choose the first resource
    assert np.array_equal(utility_vector, [100, 100, 100])
    # All choose different resources
    utility_vector = utility_matrix[:, 0, 1, 2]
    assert np.array_equal(utility_vector, [300, 300, 300])
    # The first two choose the same resource, the third chooses different
    utility_vector = utility_matrix[:, 0, 0, 1]
    assert np.array_equal(utility_vector, [150, 150, 300])


def test_utility_overflow() -> None:
    # After one step, check if there is no utility overflow for any resource
    previous_utility = np.random.randint(0, 100, size=(3, 17))
    chosen_resources = np.random.randint(0, 17, size=3)
    votes = np.random.randint(300, 500, size=17)
    is_owner = np.random.randint(0, 1, size=(3, 17))

    # Test in model with fixed-ownership
    one_day_utility = get_utility(
        previous_utility, chosen_resources, votes, is_owner, 0.8, 0.5, -1
    )
    assert (
        one_day_utility.sum(axis=0) > votes
    ).sum() == 0, f"Utility overflow detected."
    # Test in model with trashold-ownership
    one_day_utility = get_utility(
        previous_utility, chosen_resources, votes, is_owner, 0.8, 0.5, 0.35
    )
    assert (
        one_day_utility.sum(axis=0) > votes
    ).sum() == 0, f"Utility overflow detected."
