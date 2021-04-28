import chess
from typing import List

import numpy as np


class MCTSNode:
    def __init__(self, board, prior, parent, move):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        self.board: chess.Board = board
        self.move: chess.Move = move
        self.parent: MCTSNode = parent
        self.children: List[MCTSNode] = []
        self.n = 0  # times this position was visited, self.n == 0 => not expanded yet
        self.q = 0  # win ratio (wins/simulations)
        self.w = 0  # tmp value to track value output of NN
        self.p: float = prior  # prior probability, coming from NN

    def update_counts(self):
        self.n += 1
        self.q = self.w / self.n

    def get_win_ratio(self):
        return self.w / self.n

