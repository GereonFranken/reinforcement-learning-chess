import os
import subprocess
from copy import deepcopy
import chess
import chess.svg
import numpy as np
from PIL import Image


class BlobEnv:
    def __init__(self):
        self.MOVE_PENALTY = 0.05
        self.ENEMY_PENALTY = 1
        self.WIN_REWARD = 1
        self.board = chess.Board()
        self.action_space_size = self.board.legal_moves.count()
        self.BOARD_REPRESENTATION_SHAPE = (8, 8, 102)  # 8*8 board with 102 planes: see details in function doc
        self.episode_step = 0

    def reset(self):
        self.board = chess.Board()
        self.episode_step = 0
        # observation = self.get_png_board(self.board)
        return self.board

    def step(self, action, color):
        if color == self.board.turn:
            self.episode_step += 1
            self.board.push_uci(action)

            if self.board.is_checkmate():
                if ((self.board.result() == '1-0') and (color == chess.WHITE)) or \
                        ((self.board.result() == '0-1') and (color == chess.BLACK)):
                    reward = self.WIN_REWARD
                else:
                    reward = -self.ENEMY_PENALTY
            elif self.board.result() == '1/2-1/2':
                reward = 0
            else:
                reward = -self.MOVE_PENALTY

            done = False
            if reward == self.WIN_REWARD or reward == -self.ENEMY_PENALTY or \
                    reward == 0 or self.episode_step >= 120:
                done = True

            return self.board, reward, done

    def get_png_board(self, board):
        svg_name = 'board.svg'
        file = open(svg_name, 'w+')
        file.write(chess.svg.board(board))
        inkscape_path = 'D:\\Program-Files\\Inkscape\\bin\\inkscape.exe'
        subprocess.run(f'{inkscape_path} -w {self.SIZE} -h {self.SIZE} --export-type="png" {svg_name}')
        img = Image.open(os.getcwd() + '\\board.png')
        return np.array(img)

    def update_action_space(self):
        self.action_space_size = self.board.legal_moves.count()

    def get_board_representation(self, board: chess.Board, agent_color) -> np.ndarray((8, 8, 102)):
        """
        Get board representation to be fitting to the res net. Inspired but not fully like the alphazero architecture.
        The board gets transformed into binary planes for the most part. Every piece of both players gets located and a
        plane is formed. This happens for the last 8 half-moves. If the move stack is empty, planes full of zeros get
        added. Additionally there are 6 constant valued planes for the color to move, the total move count and the
        castling rights of both players to both sides.
        :param board: current board that the board representation is needed for
        :param agent_color: current player to move. Either chess.WHITE or chess.BLACK
        :return: board representation in shape (8, 8, 102). Consists of 102 planes of the 8*8 board.
            102 = (6 pieces for white + 6 pieces for black) * 8 + 6
        """
        pieces_to_represent = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        board_representation = np.empty((8, 8, 0))
        current_board = deepcopy(board)
        move_ctr = self.board.halfmove_clock + 1
        for i in range(8):
            for color in [chess.WHITE, chess.BLACK]:
                for piece in pieces_to_represent:
                    if not move_ctr:
                        board_representation = np.concatenate((board_representation, np.zeros((8, 8, 1))), axis=2)
                    else:
                        boolean_board = np.array(current_board.pieces(piece, color=color).tolist()).reshape((8, 8))
                        boolean_board = np.flip(boolean_board, axis=0) if agent_color == chess.WHITE \
                            else np.flip(boolean_board, axis=1)
                        int_board = np.expand_dims(boolean_board.astype(int), axis=2)
                        board_representation = np.concatenate((board_representation, int_board), axis=2)
                        # possibly concat plane for repetition of each player
            move_ctr = max(0, move_ctr - 1)
            if move_ctr:
                current_board.pop()
        # Additional constant valued information planes
        color_plane = np.ones((8, 8, 1)) if agent_color == chess.WHITE else np.zeros((8, 8, 1))
        move_count_plane = np.full((8, 8, 1), self.board.fullmove_number)
        white_kingside_castling_plane = np.full((8, 8, 1), self.board.has_kingside_castling_rights(chess.WHITE) * 1)
        white_queenside_castling_plane = np.full((8, 8, 1), self.board.has_queenside_castling_rights(chess.WHITE) * 1)
        black_kingside_castling_plane = np.full((8, 8, 1), self.board.has_kingside_castling_rights(chess.BLACK) * 1)
        black_queenside_castling_plane = np.full((8, 8, 1), self.board.has_queenside_castling_rights(chess.BLACK) * 1)
        board_representation = np.concatenate((board_representation,
                                               color_plane,
                                               move_count_plane,
                                               white_kingside_castling_plane,
                                               white_queenside_castling_plane,
                                               black_kingside_castling_plane,
                                               black_queenside_castling_plane), axis=2)
        # possibly concat constant valued plane for "no-progress count"
        return board_representation

