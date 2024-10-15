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
End game test
"""
initial_board = (
    ['WWWWWWWW',  # 1
     'WWWBBBBB',  # 2
     'WWWWBWBB',  # 3
     'WBWBWBWB',  # 4
     'WBBWBWWB',  # 5
     '.BBBBBWB',  # 6
     '.BBBWWBB',  # 7
     '..BBBBBB']  # 8
)

a7_board = (
    ['WWWWWWWW',  # 1
     'WWWBBWBB',  # 2
     'WWWWWWBB',  # 3
     'WBWWWBWB',  # 4
     'WBWWBWWB',  # 5
     '.WBBBBWB',  # 6
     'WWWWWWBB',  # 7
     '..BBBBBB']  # 8
)

a6_board = (
    ['WWWWWWWW',  # 1
     'WWWBBBBB',  # 2
     'WWWWBWBB',  # 3
     'WBWBWBWB',  # 4
     'WWBWBWWB',  # 5
     'WWWWWWWB',  # 6
     '.BBBWWBB',  # 7
     '..BBBBBB']  # 8
)

a8_board = (
    ['WWWWWWWW',  # 1
     'WWWBBBBB',  # 2
     'WWWWBWBB',  # 3
     'WBWBWBWB',  # 4
     'WBBWBWWB',  # 5
     '.BWBBBWB',  # 6
     '.WBBWWBB',  # 7
     'W.BBBBBB']  # 8
)

b8_board = (
    ['WWWWWWWW',  # 1
     'WWWBBBBB',  # 2
     'WWWWBWBB',  # 3
     'WWWBWBWB',  # 4
     'WWBWBWWB',  # 5
     '.WBBBBWB',  # 6
     '.WBBWWBB',  # 7
     '.WBBBBBB']  # 8
)


height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    a7_board = from_array_to_dictionary_board(a7_board)
    a6_board = from_array_to_dictionary_board(a6_board)
    a8_board = from_array_to_dictionary_board(a8_board)
    b8_board = from_array_to_dictionary_board(b8_board)
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
game_state_a7 = TwoPlayerGameState(
    game=game,
    board=a7_board,
    initial_player=player_a,
    player_max=player_b,
)

# Initialize a game state.
game_state_a6 = TwoPlayerGameState(
    game=game,
    board=a6_board,
    initial_player=player_a,
    player_max=player_b,
)

# Initialize a game state.
game_state_a8 = TwoPlayerGameState(
    game=game,
    board=a8_board,
    initial_player=player_a,
    player_max=player_b,
)

# Initialize a game state.
game_state_b8 = TwoPlayerGameState(
    game=game,
    board=b8_board,
    initial_player=player_a,
    player_max=player_b,
)

print("Given this initial state:")
print(' WWWWWW..\n',  # 1
      'BBBBBBBW\n',  # 2
      'BBWBBBBW\n',  # 3
      'BBBWBWB.\n',  # 4
      'BBWBWWBW\n',  # 5
      'BWWWWWBW\n',  # 6
      'B.WWWWW.\n',  # 7
      'BWWWW..W\n'  # 8
      )
print("This is the ranking of the possible movements: A7, A6, A8, B8")
print("1- A7  -- your score for this movement:", heuristic_1(game_state_a7),
      " <- This is the best board (so, your strategy should prefer it).")
print("2- A6  -- your score for this movement:",
      heuristic_1(game_state_a6), " <- Same score as A8.")
print("2- A8  -- your score for this movement:",
      heuristic_1(game_state_a8), " <- Same score as A6.")
print("4- B8  -- your score for this movement:", heuristic_1(game_state_b8),
      " <- This is the worst board (so, your strategy should not prefer it).")
