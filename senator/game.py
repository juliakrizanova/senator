import numpy as np
from utility import get_utility_matrix


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Regret matching algorithm"""
    positive_regret = np.maximum(regret, 0)
    sum_positive_regret = np.sum(positive_regret, axis=1)
    strategy = positive_regret / sum_positive_regret[:, None]
    return strategy


class Node:
    def __init__(self, parent, utility):
        self.parent = parent
        self.utility = utility
        self.iterations = 0

        self._num_players = utility.shape[0]
        self._num_resources = utility.shape[1]
        self._cumulative_regret = np.zeros((self._num_players, self._num_resources))
        self._cumulative_positive_regret = np.zeros((self._num_players, self._num_resources))
        self.children = {}  # key: action, value: Node

    @property
    def current_strategy(self) -> np.ndarray:
        strategy = regret_matching(self._cumulative_positive_regret)
        return strategy

    def sample_joint_action(self) -> np.ndarray:
        """Sample action for each player"""
        strategy = self.current_strategy
        actions = []
        for player in range(self._num_players):
            actions.append(np.random.choice(self._num_resources, p=strategy[player]))
        return np.array(actions)

    def _get_instaneous_regret(self, player: int, strategy: np.ndarray) -> np.ndarray:
        opponents = [i for i in range(self._num_players) if i != player]
        utility = self.utility[player]
        for opponent in reversed(opponents):  # We start from the last opponent because `np.tensordot` collapses axes
            utility = np.tensordot(utility, strategy, axes=(opponent, 0))  # TODO: check if this is correct

        on_policy_utility = np.sum(utility * strategy[player], keepdims=True)
        instantaneous_regret = utility - on_policy_utility
        return instantaneous_regret

    def rm_step(self) -> None:
        """
        Regret matching plus step. For each player, we compute his regret and compute the new strategy profile.
        """
        self.iterations += 1
        strategy = self.current_strategy
        for player in range(self._num_players):

            instantaneous_regret = self._get_instaneous_regret(player, strategy)

            self._cumulative_regret[player] = self._cumulative_regret[player] + instantaneous_regret

    def rm_plus_step(self) -> None:
        """
        Regret matching plus step. For each player, we compute his regret and compute the new strategy profile.
        """
        self.iterations += 1
        strategy = self.current_strategy
        player = self.iterations % self._num_players

        instantaneous_regret = self._get_instaneous_regret(player, strategy)

        self._cumulative_positive_regret[player] = np.maximum(
                                self._cumulative_positive_regret[player] + instantaneous_regret, 0)

    def nash_conv(self) -> float:
        """
        Compute the Nash Convexity of the current node.
        """
        strategy = self.current_strategy
        nash_convexity = 0
        for player in range(self._num_players):
            instantaneous_regret = self._get_instaneous_regret(player, strategy)
            nash_convexity += np.sum(instantaneous_regret * strategy[player])

        return nash_convexity


class Game:
    def __init__(self):
        root_utility = get_utility_matrix()  # TODO: see what it takes to get the utility matrix
        self.root = Node(parent=None, utility=root_utility)

    def add_child(self, node: Node, joint_action: np.ndarray) -> None:
        child_utility = get_utility_matrix()  # TODO: see what it takes to get the utility matrix
        node.children[tuple(joint_action)] = Node(parent=node, utility=child_utility)

    def solve_node_via_rm_plus(self, node: Node, iterations: int = 1000) -> None:
        for _ in range(iterations):
            node.rm_plus_step()

        if node.nash_conv() <= 0.001:
            print(f"Nash Convexity is {node.nash_conv}, something is wrong...")

    def run_greedy_search(self, num_expansions: int = 10):
        """
        Run a greedy search algorithm, periodically expanding the tree. Each expansion is done by first solving the
        node using the RM+ algorithm, sampling a jjoing action, and adding the child to the tree.
        TODO: decide how to do the expanding. Baseline is to just do a path of length 5.
        """




