import numpy as np
OWNER_TRASHOLD = 0.35
FILE_PATH = "data.csv"

def get_utility(previous_utility: np.ndarray, current_strategy_profile: np.ndarray, 
                votes: np.ndarray) -> np.ndarray:
    num_players, num_resources = previous_utility.shape
    
    current_utility = np.zeros((num_players, num_resources))
    
    
    is_owner = previous_utility >= OWNER_TRASHOLD * votes
    current_utility = (previous_utility / 2) * is_owner
    remaining_votes = votes - np.sum(current_utility, axis=0)
    
    strategy_sum = np.sum(current_strategy_profile, axis=0)
    strategy_sum = np.where(strategy_sum == 0, 1, strategy_sum)
    
    current_utility += (remaining_votes * current_strategy_profile.T / strategy_sum).T
    
    return current_utility



def load_data(file_path: str) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter=";", skiprows=1)
    return data

def parse_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
    votes = data[:, 1]
    initial_utility = data[:, [3,4,5]].T
    return votes, initial_utility

