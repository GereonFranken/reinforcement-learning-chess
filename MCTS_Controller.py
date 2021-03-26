import numpy as np
import chess
from numpy.random import sample
from multiprocessing.dummy import Pool, Manager
from copy import deepcopy
import random

from tensorflow.python.autograph.operators import int_

from Board_Evaluation import evaluate_position


class MCTSController(object):

    def __init__(self, manager, c=1.5):
        super().__init__()

        self.visits = manager.dict()
        self.differential = manager.dict()
        self.C = c

    def record(self, board: chess.Board, score):
        self.visits["total"] = self.visits.get("total", 1) + 1
        self.visits[hash(board.fen())] = self.visits.get(hash(board.fen()), 0) + 1
        self.differential[hash(board.fen())] = self.differential.get(hash(board.fen()), 0) + score

    r"""
    Runs a single, random heuristic guided playout starting from a given state. This updates the 'visits' and 'differential'
    counts for that state, as well as likely updating many children states.
    """

    def playout(self, board: chess.Board, expand=10):

        if expand == 0 or board.is_game_over():
            score = evaluate_position(board)
            self.record(board, score)
            # print ('X' if board.turn==1 else 'O', score)
            return score

        action_mapping = {}

        for move in list(board.legal_moves):
            board.push(move)
            action_mapping[move] = self.heuristic_value(board)
            board.pop()

        chosen_action = max(action_mapping, key=action_mapping.get) # replace with priors from NN
        board.push(chosen_action)
        score = -self.playout(board, expand=expand - 1)  # play branch
        board.pop()
        self.record(board, score)
        return score

    r"""
    Evaluates the "value" of a state as a bandit problem, using the value + exploration heuristic.
    """

    def heuristic_value(self, board: chess.Board):
        simulated_game_states = self.visits.get("total", 1)
        number_move_played_in_simulation = self.visits.get(hash(board.fen()), 1e-9)
        heuristic_score = self.differential.get(hash(board.fen()), 0) * 1.0 / number_move_played_in_simulation
        return heuristic_score + self.C * (np.log(simulated_game_states) / number_move_played_in_simulation)

    r"""
    Evaluates the "value" of a state by randomly playing out boards starting from that state and noting the win/loss ratio.
    """

    def value(self, board: chess.Board, playouts=10, steps=5):

        # play random playouts starting from that board value
        with Pool() as p:
            scores = p.map(self.playout, [deepcopy(board) for i in range(0, playouts)])

        return self.differential[hash(board.fen())] * 1.0 / self.visits[hash(board.fen())]

    r"""
    Chooses the move that results in the highest value state.
    """

    def best_move(self, board: chess.Board, playouts=20):

        action_mapping = {}

        for action in list(board.legal_moves):
            board.push(action)
            action_mapping[action] = self.value(board, playouts=playouts)
            board.pop()

        print({a: "{0:.2f}".format(action_mapping[a]) for a in action_mapping})
        return max(action_mapping, key=action_mapping.get)

# board = chess.Board()
# for i in range(20):
#     board.push(random.choice(list(board.legal_moves)))
# tree = MCTSController(Manager())
# print(board)
# from BlobEnv import BlobEnv
# blob = BlobEnv()
# blob.get_png_board(board)
# best_move = tree.best_move(board)
# print(best_move)

from BlobEnv import BlobEnv
env = BlobEnv()
# for i in range(20):
#     env.board.push(random.choice(list(env.board.legal_moves)))
env.get_board_representation(chess.WHITE)



