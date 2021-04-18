import chess
from typing import List


class MCTSNode:
    def __init__(self, board, prior, parent):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        self.board: chess.Board = board
        self.parent: MCTSNode = parent
        self.n = 0  # times this position was visited, self.n == 0 => not expanded yet
        self.q = 0  # win ratio (wins/simulations)
        self.w = 0  # tmp value to track value output of NN
        self.p = prior  # prior probability, coming from NN
        self.children: List[MCTSNode] = []

    def update_counts(self):
        self.n += 1
        self.q = self.w / self.n

    def get_win_ratio(self):
        return self.w / self.n

