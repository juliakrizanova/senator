import numpy as np

RESOURCES = 17
PLAYERS = 3
FILE_PATH = "data.csv"


def get_utility(
    previous_utility: np.ndarray,
    chosen_resources: list[int],
    votes: np.ndarray,
    is_owner: np.ndarray = np.zeros((PLAYERS, RESOURCES)),
    owner_loss: float = 0.8,
    other_loss: float = 0,
    owner_trashold: float = -1,
) -> np.ndarray:

    num_players, num_resources = previous_utility.shape

    current_strategy_profile = np.zeros((num_players, num_resources))
    current_strategy_profile[np.arange(num_players), chosen_resources] = 1

    current_utility = np.zeros((num_players, num_resources))

    if owner_trashold <= 1.0:
        is_owner = (
            previous_utility >= owner_trashold * votes
        )  # original variant of the model, where is_owner is computed by using a trashold

    current_utility = ((previous_utility * owner_loss) * is_owner) + (
        previous_utility * other_loss
    ) * (1 - is_owner)

    remaining_votes = votes - np.sum(current_utility, axis=0)

    strategy_sum = np.sum(current_strategy_profile, axis=0)
    strategy_sum = np.where(strategy_sum == 0, 1, strategy_sum)

    current_utility += (
        remaining_votes[:, None] * current_strategy_profile.T / strategy_sum[:, None]
    ).T

    return current_utility


def get_utility_matrix(
    previous_utility: np.ndarray,
    votes: np.ndarray,
    is_owner: np.ndarray,
    owner_loss: float = 0.8,
    other_loss: float = 0.0,
    owner_trashold: float = -1,
) -> np.ndarray:

    utility_matrix = np.zeros((PLAYERS, RESOURCES, RESOURCES, RESOURCES))

    for i in range(RESOURCES):
        for j in range(RESOURCES):
            for k in range(RESOURCES):
                utility_matrix[:, i, j, k] = get_utility(
                    previous_utility,
                    [i, j, k],
                    votes,
                    is_owner,
                    owner_loss,
                    other_loss,
                    owner_trashold,
                ).sum(axis=1)

    return utility_matrix


def load_data(file_path: str = FILE_PATH) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    return data


def parse_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    votes = data[:, 1]
    initial_utility = data[:, [2, 3, 4]].T
    is_owner = data[:, [5, 6, 7]].T
    return votes, initial_utility, is_owner
