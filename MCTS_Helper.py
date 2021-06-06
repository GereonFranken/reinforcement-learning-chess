import numpy as np
import chess


LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8']
PROMOTED_TO = ['q', 'r', 'b', 'n']


class MCTSHelper:

    def __init__(self):
        self.uci_labels = self._get_uci_mapping()
        self.empty_board: np.ndarray = np.array([])
        self.main_diag = []
        self.second_diag = []

    def __new__(cls):  # make class singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(MCTSHelper, cls).__new__(cls)
        return cls.instance

    def _get_uci_mapping(self):
        """
        :return: A list of all possible moves on a chessboard in uci format. List length = 1968.
        """
        possible_uci_moves = []
        all_squares: np.ndarray = np.array([letter + number for number in NUMBERS for letter in LETTERS])
        self.empty_board = all_squares.reshape((8, 8))
        self.main_diag = np.diagonal(self.empty_board)
        self.second_diag = np.diagonal(np.flipud(self.empty_board))
        for pick_up_square in all_squares:
            bishop_moves = self._get_bishop_moves(pick_up_square)
            knight_moves = self._get_knight_moves(pick_up_square, all_squares)
            for place_down_square in all_squares:
                potential_move: str = pick_up_square + place_down_square
                # horizontal and vertical moves: Either move is in one column (same letter) or one row (same number)
                if ((potential_move.count(pick_up_square[0]) == 2) ^ (potential_move.count(pick_up_square[1]) == 2)) \
                        or place_down_square in bishop_moves \
                        or place_down_square in knight_moves:
                    possible_uci_moves.append(potential_move)
        possible_uci_moves.extend(self._get_promotion_moves())
        return np.array(possible_uci_moves)

    def _get_bishop_moves(self, square):
        """
        first calculate the offsets of the diagonals by checking the number of rows between the square and the main
        diagonal. Example: square = b6 and the ascii value of b is 98. Search for the 6 in the main diagonal and get the
        square which will be d6. Subtract the ascii value of the pick up square from the main diag square letter
        (d = 100) and you get and diagonal offset of -2. This is exactly the offset you need to get the first possible
        diagonal for the square. Do the same with a flipped board to get the second one.
        :param square: pick up square
        :return: the two diagonals which you reach from the pick up square
        """
        offset_diag1 = ord(square[0]) - ord(
            self.main_diag[np.flatnonzero(np.core.defchararray.find(self.main_diag, square[1]) != -1)][0][0])
        offset_diag2 = ord(square[0]) - ord(
            self.second_diag[np.flatnonzero(np.core.defchararray.find(self.second_diag, square[1]) != -1)][0][0])
        diag1 = np.diagonal(self.empty_board, offset=offset_diag1)
        diag2 = np.diagonal(np.flipud(self.empty_board), offset=offset_diag2)
        diag1 = diag1[diag1 != square]
        diag2 = diag2[diag2 != square]
        return np.concatenate((diag1, diag2))

    @staticmethod
    def _get_knight_moves(square, all_squares):
        """
        :param square: pick up square
        :param all_squares: entire empty board
        :return: All possible target squares for knight moves for a given pick up square
        """
        possible_moves: np.ndarray = np.array([])
        operations = [[2, 1], [-2, -1]]
        for operation in operations:
            for _ in range(2):
                new_set = [chr(ord(square[0]) + operation[0]) + str(int(square[1]) + operation[1] * j)
                           for j in (-1, +1)]
                possible_moves = np.concatenate((possible_moves, new_set))
                operation.reverse()
        boolean_filter = [np.any(all_squares == move) for move in possible_moves]
        return possible_moves[boolean_filter]

    @staticmethod
    def _get_promotion_moves():
        """
        :return: All promotion moves in uci format
        """
        promotion_moves = []
        for i in LETTERS:
            for j in [(2, -1), (7, 1)]:
                pick_up_square = i + str(j[0])
                for piece in PROMOTED_TO:
                    promotion_moves.extend([
                        pick_up_square + i + str(j[0] + j[1]) + piece,
                        pick_up_square + chr(ord(i) - 1) + str(j[0] + j[1]) + piece
                        if chr(ord(i) - 1) in LETTERS else None,
                        pick_up_square + chr(ord(i) + 1) + str(j[0] + j[1]) + piece
                        if chr(ord(i) + 1) in LETTERS else None,
                    ])
        return np.array(list(filter(None, promotion_moves)))

    def transform_priors(self, board: chess.Board, priors: []):
        """
        Transforms the list of all priors for all moves and map them to the legal moves in the current position
        :param board: current position
        :param priors: list of all priors
        :return: List of (move, prior) tuples. One prior for each legal move
        """
        legal_priors = []
        legal_moves = []
        for legal_move in list(map(lambda move: move.uci(), list(board.legal_moves))):
            i: int = np.where(self.uci_labels == legal_move)[0][0]
            legal_moves.append(legal_move)
            legal_priors.append(np.array(priors).reshape(1968)[i])
        legal_priors = [prior / np.sum(np.array(legal_priors)) for prior in legal_priors]  or 1 # normalize
        return np.array([(legal_moves[j], legal_priors[j]) for j in range(len(legal_priors))])

    def find_index_of_move(self, move: str) -> int:
        return np.argwhere(self.uci_labels == move)[0][0]

    def mask_probabilites(self, moves, probabilites):
        mask = np.zeros((1968))
        for i, move in enumerate(moves):
            mask[self.find_index_of_move(move)] = probabilites[i]
        return mask

