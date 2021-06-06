import numpy as np
import chess
from BlobEnv import BlobEnv
from tensorflow.keras import Model
from MCTS_Node import MCTSNode
from MCTS_Helper import MCTSHelper
from copy import deepcopy
from typing import List, TypeVar


class MCTSController:

    def __init__(self, board: chess.Board = None, model: Model = None):
        self.root = MCTSNode(board, None, None, None)
        self.current_node: MCTSNode = self.root  # current node that is looked at. Is root node at the beginning
        self.helper = MCTSHelper()
        self.env = BlobEnv()
        self.model = model
        self.c = 1.2  # exploration param
        self.t = 0.8  # how greedily should an action be chosen according to the visit count

    def selection(self, node_children: List[MCTSNode]):
        # add Dirichlet noise to priors
        noises = np.random.dirichlet(np.full((len(node_children)), 0.3))  # 0.3 as Dirichlet alpha
        for i, child in enumerate(node_children):
            child.p += noises[i]
        total_visits = np.sum(list(map(lambda child: child.n, node_children)))
        move_scores = [child.q + self.c * child.p * (np.sqrt(total_visits) / (1 + child.n))
                       for child in node_children]
        return node_children[np.argmax(move_scores)]

    def expand(self, node: MCTSNode):
        board_representation = self.env.get_board_representation(node.board, node.board.turn)
        priors, value = self.model.predict(np.expand_dims(board_representation, axis=0))
        priors: List[(str, float)] = self.helper.transform_priors(node.board, priors)
        for move, prior in priors:
            new_board = deepcopy(node.board)
            new_board.push_uci(move)
            node.children.append(MCTSNode(new_board, float(prior), node, move))
        self.backpropagation(node, value[0], node.board.turn)

    def backpropagation(self, node: MCTSNode, value, leaf_turn: chess.WHITE or chess.BLACK):
        if node.parent is not None:
            node.w = node.w - value if node.board.turn != leaf_turn else node.w + value
            node.update_counts()
            self.backpropagation(node.parent, value, leaf_turn)

    def choose_best_action(self):
        # choose best child according to visit count
        total_visits = np.sum(list(map(lambda child: child.n, self.root.children)))
        return self.root.children[np.argmax([(child.n ** (1 / self.t)) / (total_visits ** (1 / self.t))
                                             for child in self.root.children])]

    def find_best_move(self):
        for _ in range(30):
            self.expand(self.current_node)
            self.current_node = self.root
            while len(self.current_node.children):
                self.current_node = self.selection(self.current_node.children)
        moves = [child.move for child in self.root.children]
        move_probabilities = [child.p for child in self.root.children]
        normalized_probabilities = [np.divide(prob, np.sum(move_probabilities)) for prob in move_probabilities]
        masked_probabilites = self.helper.mask_probabilites(moves, normalized_probabilities)
        return self.choose_best_action(), masked_probabilites
