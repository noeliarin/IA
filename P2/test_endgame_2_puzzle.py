"""Test case for an heuristic.

Authors:
    Alejandro Bellogin <alejandro.bellogin@uam.es>
    Daniel Fernandez <daniel.fernandezs@uam.es>

"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from game import Player, TwoPlayerGameState
from heuristic import Heuristic
from reversi import (
    Reversi,
    from_array_to_dictionary_board,
)
from strategy import (
    MinimaxStrategy,
)


def subtraction_heuristic(state: TwoPlayerGameState) -> float:

    if state.end_of_game:
        scores = state.scores
    else:
        scores = [
            state.game._player_coins(state.board, state.next_player.label),
            state.game._player_coins(
                state.board, state.game.opponent(state.next_player).label)
        ]

    # Evaluation of the state from the point of view of MAX
    assert isinstance(scores, (Sequence, np.ndarray))
    score_difference = scores[0] - scores[1]

    if state.is_player_max(state.player1):
        state_value = score_difference
    elif state.is_player_max(state.player2):
        state_value = - score_difference
    else:
        raise ValueError('Player MAX not defined')

    return state_value


def heuristic_1(state: TwoPlayerGameState) -> float:
    return subtraction_heuristic(state)


player_minimax_1 = Player(
    name='Black',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='heuristic_1_B',
                            evaluation_function=heuristic_1),
        max_depth_minimax=3,
        verbose=0,
    ),
)

player_minimax_2 = Player(
    name='White',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='heuristic_1_W',
                            evaluation_function=heuristic_1),
        max_depth_minimax=3,
        verbose=0,
    ),
)


# minimax vs minimax player
player_a, player_b = player_minimax_1, player_minimax_2

"""
Endgame test
"""
initial_board = (
    ['BBBBWWWB',  # 1
     'WBBBWWWB',  # 2
     'WBBWBBWB',  # 3
     'WBBBWBWB',  # 4
     'WWBWWWWB',  # 5
     'WBWWWWWB',  # 6
     'WWBWWWW.',  # 7
     'WWWWW...']  # 8
)

g8_board = (
    ['BBBBWWWB',  # 1
     'WBBBWWWB',  # 2
     'WBBWBBWB',  # 3
     'WBBBWBWB',  # 4
     'WWBBWWWB',  # 5
     'WBWWBWWB',  # 6
     'WWBWWBW.',  # 7
     'WWWWW.B.']  # 8
)

h7_board = (
    ['BBBBWWWB',  # 1
     'WBBBWWWB',  # 2
     'WBBBBBWB',  # 3
     'WBBBBBWB',  # 4
     'WWBWWBWB',  # 5
     'WBWWWWBB',  # 6
     'WWBBBBBB',  # 7
     'WWWWW...']  # 8
)

h8_board = (
    ['BBBBWWWB',  # 1
     'WBBBWWWB',  # 2
     'WBBWBBWB',  # 3
     'WBBBWBWB',  # 4
     'WWBWBWWB',  # 5
     'WBWWWBWB',  # 6
     'WWBWWWB.',  # 7
     'WWWWW..B']  # 8
)

f8_board = (
    ['BBBBWWWB',  # 1
     'WBBBWWWB',  # 2
     'WBBWBBWB',  # 3
     'WBBBWBWB',  # 4
     'WWBWWBWB',  # 5
     'WBWBWBWB',  # 6
     'WWBWBBB.',  # 7
     'WWWWWB..']  # 8
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    g8_board = from_array_to_dictionary_board(g8_board)
    h7_board = from_array_to_dictionary_board(h7_board)
    h8_board = from_array_to_dictionary_board(h8_board)
    f8_board = from_array_to_dictionary_board(f8_board)
except ValueError:
    raise ValueError('Wrong configuration of the board')
else:
    print("Successfully initialised board from array")


# Initialize a reversi game.
game = Reversi(
    player1=player_a,
    player2=player_b,
    height=height,
    width=width,
)

# Initialize a game state.
game_state_g8 = TwoPlayerGameState(
    game=game,
    board=g8_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_h7 = TwoPlayerGameState(
    game=game,
    board=h7_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_h8 = TwoPlayerGameState(
    game=game,
    board=h8_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_f8 = TwoPlayerGameState(
    game=game,
    board=f8_board,
    initial_player=player_b,
    player_max=player_a,
)


print("Given this initial state:")
print(' BBBBWWWB\n',  # 1
      'WBBBWWWB\n',  # 2
      'WBBWBBWB\n',  # 3
      'WBBBWBWB\n',  # 4
      'WWBWWWWB\n',  # 5
      'WBWWWWWB\n',  # 6
      'WWBWWWW.\n',  # 7
      'WWWWW...\n'  # 8
      )
print("This is the ranking of the possible movements: G8, H7, H8, F8")
print("1- G8  -- your score for this movement:", heuristic_1(game_state_g8),
      " <- This is the best board (so, your strategy should prefer it).")
print("2- H7  -- your score for this movement:", heuristic_1(game_state_h7))
print("3- H8  -- your score for this movement:", heuristic_1(game_state_h8))
print("4- F8  -- your score for this movement:", heuristic_1(game_state_f8),
      " <- This is the worst board (so, your strategy should not prefer it).")
