import numpy as np
from senator.utility import get_utility_matrix, get_utility


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Regret matching algorithm"""
    positive_regret = np.maximum(regret, 0)
    sum_positive_regret = np.sum(positive_regret, axis=1)
    strategy = positive_regret / sum_positive_regret[:, None]
    return np.where(np.isnan(strategy), 1 / strategy.shape[1], strategy)


class Node:
    def __init__(self, parent, utility_matrix: np.ndarray) -> None:

        self.parent = parent
        self.utility_matrix = utility_matrix
        self.iterations = 0

        self._num_players = utility_matrix.shape[0]
        self._num_resources = utility_matrix.shape[1]
        self._cumulative_regret = np.zeros((self._num_players, self._num_resources))
        self._cumulative_positive_regret = np.zeros(
            (self._num_players, self._num_resources)
        )
        self.children: dict[tuple, Node] = {}  # key: action, value: Node

    @property
    def current_strategy(self) -> np.ndarray:
        strategy = regret_matching(self._cumulative_regret)
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
        utility_matrix = self.utility_matrix[player]
        for opponent in reversed(
            opponents
        ):  # We start from the last opponent because `np.tensordot` collapses axes
            utility_matrix = np.tensordot(
                utility_matrix, strategy[opponent], axes=([opponent], [0])
            )  # TODO: check if this is correct

        on_policy_utility = np.sum(utility_matrix * strategy[player], keepdims=True)
        instantaneous_regret = utility_matrix - on_policy_utility
        return instantaneous_regret

    def rm_step(self) -> None:
        """
        Regret matching step. For each player, we compute his regret and add it to the cumulative regret.
        """
        self.iterations += 1
        strategy = self.current_strategy
        for player in range(self._num_players):

            instantaneous_regret = self._get_instaneous_regret(player, strategy)

            self._cumulative_regret[player] = (
                self._cumulative_regret[player] + instantaneous_regret
            )

    def rm_plus_step(self) -> None:
        """
        Regret matching plus step. For each player, we compute his regret and add it to the cumulative positive regret.
        """

        strategy = self.current_strategy
        player = self.iterations % self._num_players
        self.iterations += 1

        instantaneous_regret = self._get_instaneous_regret(player, strategy)

        self._cumulative_positive_regret[player] = np.maximum(
            self._cumulative_positive_regret[player] + instantaneous_regret, 0
        )

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
    def __init__(
        self,
        initial_utility: np.ndarray,
        votes: np.ndarray,
        is_owner: np.ndarray,
        num_iterations: int = 7,
        owner_loss: float = 1,
        other_loss: float = 0,
        owner_trashold: float = 0,
    ) -> None:

        self.num_iterations = num_iterations
        self.owner_loss = owner_loss
        self.other_loss = other_loss
        self.owner_trashold = owner_trashold
        self.is_owner = is_owner
        self.votes = votes
        self.initial_utility = initial_utility
        root_utility_matrix = get_utility_matrix(initial_utility, votes, is_owner)
        self.root = Node(parent=None, utility_matrix=root_utility_matrix)

    def add_child(
        self,
        node: Node,
        joint_action: np.ndarray,
        current_utility: np.ndarray,
        votes: np.ndarray,
    ) -> None:
        child_utility_matrix = get_utility_matrix(
            current_utility,
            votes,
            self.is_owner,
            self.owner_loss,
            self.other_loss,
            self.owner_trashold,
        )  # TODO: see what it takes to get the utility matrix
        node.children[tuple(joint_action)] = Node(
            parent=node, utility_matrix=child_utility_matrix
        )

    def solve_node_via_rm_plus(self, node: Node, iterations: int = 1000) -> None:
        for _ in range(iterations):
            node.rm_plus_step()

        if (conv := node.nash_conv()) >= 0.001:
            print(f"Nash Convexity is {conv}, something is wrong...")

    def solve_node_via_rm(self, node: Node, iterations: int = 1000) -> None:
        for _ in range(iterations):
            node.rm_step()

        if (conv := node.nash_conv()) >= 0.001:
            print(f"Nash Convexity is {conv}, something is wrong...")

    def run_greedy_search(self, num_iterations: int = 7) -> tuple:

        actions_in_steps = []
        strategies_in_steps = []

        current_node = self.root
        utility = self.initial_utility

        # print(f"Initial utility\nu: {utility.sum(axis=1)}")

        for i in range(num_iterations):
            self.solve_node_via_rm(current_node)
            strategies_in_steps.append(current_node.current_strategy)
            joint_action = current_node.sample_joint_action()

            actions_in_steps.append(joint_action)

            utility = get_utility(
                utility,
                joint_action,
                self.votes,
                self.is_owner,
                self.owner_loss,
                self.other_loss,
                self.owner_trashold,
            )
            self.add_child(current_node, joint_action, utility, self.votes)
            current_node = current_node.children[tuple(joint_action)]

            # print(f"DAY {i} \na: {joint_action} \nu: {np.floor(utility.sum(axis=1))}")
        return actions_in_steps, utility, strategies_in_steps
