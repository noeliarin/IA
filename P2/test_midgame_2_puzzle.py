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
Midgame test
"""
initial_board = (
    ['.WWW....',  # 1
     '.WWWB...',  # 2
     'BWWBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     '.WWBB...',  # 7
     '.WWWWWW.']  # 8
)

a8_board = (
    ['.WWW....',  # 1
     '.WWWB...',  # 2
     'BWWBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     '.BWBB...',  # 7
     'BWWWWWW.']  # 8
)

e1_board = (
    ['.WWWB...',  # 1
     '.WWBB...',  # 2
     'BWBBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     '.WWBB...',  # 7
     '.WWWWWW.']  # 8
)

a1_board = (
    ['BWWW....',  # 1
     '.BWWB...',  # 2
     'BWBBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     '.WWBB...',  # 7
     '.WWWWWW.']  # 8
)

a2_board = (
    ['.WWW....',  # 1
     'BBBBB...',  # 2
     'BBWBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     '.WWBB...',  # 7
     '.WWWWWW.']  # 8
)

a7_board = (
    ['.WWW....',  # 1
     '.WWWB...',  # 2
     'BWWBB...',  # 3
     'BBBBBB..',  # 4
     'BBBBBB..',  # 5
     'BBBBBB..',  # 6
     'BBBBB...',  # 7
     '.WWWWWW.']  # 8
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    a8_board = from_array_to_dictionary_board(a8_board)
    e1_board = from_array_to_dictionary_board(e1_board)
    a1_board = from_array_to_dictionary_board(a1_board)
    a2_board = from_array_to_dictionary_board(a2_board)
    a7_board = from_array_to_dictionary_board(a7_board)
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
game_state_a8 = TwoPlayerGameState(
    game=game,
    board=a8_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_e1 = TwoPlayerGameState(
    game=game,
    board=e1_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_a1 = TwoPlayerGameState(
    game=game,
    board=a1_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_a2 = TwoPlayerGameState(
    game=game,
    board=a2_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_a7 = TwoPlayerGameState(
    game=game,
    board=a7_board,
    initial_player=player_b,
    player_max=player_a,
)


print("Given this initial state:")
print(' .WWW....\n',  # 1
      '.WWWB...\n',  # 2
      'BWWBB...\n',  # 3
      'BBBBBB..\n',  # 4
      'BBBBBB..\n',  # 5
      'BBBBBB..\n',  # 6
      '.WWBB...\n',  # 7
      '.WWWWWW.\n'  # 8
      )
print("This is the ranking of the possible movements: A8, E1, A1, A2, A7,")
print("1- A8  -- your score for this movement:", heuristic_1(game_state_a8),
      " <- This is the best board (so, your strategy should prefer it).")
print("2- E1  -- your score for this movement:", heuristic_1(game_state_e1))
print("3- A1  -- your score for this movement:", heuristic_1(game_state_a1))
print("4- A2  -- your score for this movement:", heuristic_1(game_state_a2))
print("5- A7  -- your score for this movement:", heuristic_1(game_state_a7),
      " <- This is the worst board (so, your strategy should not prefer it).")
