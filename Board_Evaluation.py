import chess
import numpy as np

PIECE_VALUES = {
    'K': 200,
    'Q': 9,
    'R': 5,
    'N': 3,
    'B': 3,
    'P': 1,
    'k': -200,
    'q': -9,
    'r': -5,
    'n': -3,
    'b': -3,
    'p': -1,
}


def evaluate_position(position: chess.Board):
    material = 0
    board_symbols = [position.piece_map()[i].symbol() for i in position.piece_map()]
    material += np.sum([PIECE_VALUES[symbol] for symbol in board_symbols])

    king_safety_score = np.sum([
        calculate_king_safety_score({piece[1].symbol(): piece}, position)
        for piece in position.piece_map().items()
        if piece[1].symbol() == 'k' or piece[1].symbol() == 'K'
    ])

    mobility = get_player_mobility_score(position)

    return material + king_safety_score + mobility


def calculate_king_safety_score(piece_player: dict, position: chess.Board):
    if 'K' in piece_player.keys():
        get_virtual_king_mobility(piece_player['K'], position)
        white_open_file = get_open_file_score(piece_player['K'], position)
        white_pawn_shield = check_pawn_shield(piece_player['K'], position)
        white_virtual_mobility = get_virtual_king_mobility(piece_player['K'], position)
        return white_open_file + white_pawn_shield + white_virtual_mobility
    else:
        black_open_file = get_open_file_score(piece_player['k'], position)
        black_pawn_shield = check_pawn_shield(piece_player['k'], position)
        black_virtual_mobility = get_virtual_king_mobility(piece_player['k'], position)
        return black_open_file + black_pawn_shield + black_virtual_mobility


def get_open_file_score(piece: tuple, position: chess.Board):
    fen_position = position.fen().split()[0]
    danger_multiplier = 1
    piece_file = piece[0] % 8
    open_files_ctr = 0
    turn = 1 if piece[1].symbol() == 'K' else -1
    is_file_open = [True, True, True]
    if piece[1].symbol() == 'k':
        danger_multiplier += fen_position.count('q') * 0.5
        danger_multiplier += fen_position.count('r') * 0.25
    else:
        danger_multiplier += fen_position.count('Q') * 0.5
        danger_multiplier += fen_position.count('R') * 0.25
    for board_piece in position.piece_map().items():
        if (board_piece[0] % 8 == piece_file or
            board_piece[0] % 8 == piece_file - 1 or
            board_piece[0] % 8 == piece_file + 1) and \
                (board_piece[1].symbol() != 'k' and
                 board_piece[1].symbol() != 'K'):
            is_file_open[board_piece[0] % 8 - piece_file + 1] = False
            if not any(is_file_open):
                break
    for file_open in is_file_open:
        if file_open:
            open_files_ctr += 1
    return round(danger_multiplier * open_files_ctr * turn, 3)


def check_pawn_shield(piece: tuple, position: chess.Board):
    """
    Note: Only pawn shields in the initial direction of the opponent are considered. Pawn shields in the end game where
    the king is positioned differently and the pawns might be on the side of the king or even behind are not seen.
    :param piece: contains board position and string symbol
    :param position: current board position
    :return: pawn shield score
    """
    turn = 1 if piece[1].symbol() == 'K' else -1
    pawn_shield_ctr = 0
    if piece[1].symbol() == 'K':
        try:
            if piece[0] % 8 == 0:
                if position.piece_map()[piece[0]+8].symbol() == 'P':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]+9].symbol() == 'P':
                    pawn_shield_ctr += 1
            elif piece[0] % 8 == 7:
                if position.piece_map()[piece[0]+7].symbol() == 'P':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]+8].symbol() == 'P':
                    pawn_shield_ctr += 1
            else:
                if position.piece_map()[piece[0]+7].symbol() == 'P':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]+8].symbol() == 'P':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]+9].symbol() == 'P':
                    pawn_shield_ctr += 1
        except KeyError:
            pass
    if piece[1].symbol() == 'k':
        try:
            if piece[0] % 8 == 0:
                if position.piece_map()[piece[0]-8].symbol() == 'p':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]-7].symbol() == 'p':
                    pawn_shield_ctr += 1
            elif piece[0] % 8 == 7:
                if position.piece_map()[piece[0]-8].symbol() == 'p':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]-9].symbol() == 'p':
                    pawn_shield_ctr += 1
            else:
                if position.piece_map()[piece[0]-7].symbol() == 'p':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]-8].symbol() == 'p':
                    pawn_shield_ctr += 1
                if position.piece_map()[piece[0]-9].symbol() == 'p':
                    pawn_shield_ctr += 1
        except KeyError:
            pass
    return turn * pawn_shield_ctr


def get_virtual_king_mobility(piece: tuple, position: chess.Board):
    """
    calculate virtual mobility score of the king to get a measurement of the kings freedom
    For that, replace the king with a queen and see how many legal moves the queen would have before colliding with an
    enemy piece (disregarding which piece). This gives a feeling for x-ray attacks or other dangers that might come in
    the near future. Since its not very accurate, its normalized with the max number of legal moves of a queen, so that
    the return value is always between 0 and |1|
    """
    turn = 1 if piece[1].symbol() == 'K' else -1
    position_copy = position
    position_copy = position_copy.fen().split(' ')[0].replace('Q', 'P').replace('K', 'Q')
    board_copy = (chess.Board(position_copy))
    for square in board_copy.piece_map():
        if board_copy.piece_map()[square].symbol().isupper() and board_copy.piece_map()[square].symbol() != 'Q':
            board_copy.remove_piece_at(square)
    return len(list(board_copy.legal_moves)) / 27 * turn


def get_player_mobility_score(position: chess.Board):
    """
    calculate the mobility score of the players, meaning the amount of your legal moves compared to the amount
    of the opponents legal moves
    :param position: current board position
    :return: mobility score: the formula for it is self-constructed according to gut-feeling. Adapt it if needed
    """
    if position.turn:
        white_mobility = len(list(position.legal_moves))
        position.turn = not position.turn
        black_mobility = len(list(position.legal_moves))
        return np.tanh(0.5 * (white_mobility - black_mobility)) * 3
    else:
        black_mobility = len(list(position.legal_moves))
        position.turn = not position.turn
        white_mobility = len(list(position.legal_moves))
        return np.tanh(0.5 * (white_mobility - black_mobility)) * 3

