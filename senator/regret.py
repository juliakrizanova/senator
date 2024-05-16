import numpy as np

def solve_matrix_game(utility_matrix: np.ndarray) -> np.ndarray:
    num_players, num_resources = utility_matrix.shape
    strategy_profile = np.zeros((num_players, num_resources))
    
    strategy_profile = utility_matrix / np.sum(utility_matrix, axis=0)
    
    return strategy_profile
