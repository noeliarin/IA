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
Opening test
"""
initial_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BWW..',  # 4
     '...BW...',  # 5
     '...WB...',  # 6
     '........',  # 7
     '........']  # 8
)

e3_board = (  # This is the best option
    ['........',  # 1
     '........',  # 2
     '...BB...',  # 3
     '...BBW..',  # 4
     '...BB...',  # 5
     '...WB...',  # 6
     '........',  # 7
     '........']  # 8
)

f3_board = (
    ['........',  # 1
     '........',  # 2
     '...B.B..',  # 3
     '...BBW..',  # 4
     '...BW...',  # 5
     '...WB...',  # 6
     '........',  # 7
     '........']  # 8
)

g4_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BBBB.',  # 4
     '...BW...',  # 5
     '...WB...',  # 6
     '........',  # 7
     '........']  # 8
)

f5_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BBW..',  # 4
     '...BBB..',  # 5
     '...WB...',  # 6
     '........',  # 7
     '........']  # 8
)

c6_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BWW..',  # 4
     '...BW...',  # 5
     '..BBB...',  # 6
     '........',  # 7
     '........']  # 8
)

f6_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BWW..',  # 4
     '...BB...',  # 5
     '...WBB..',  # 6
     '........',  # 7
     '........']  # 8
)

d7_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BWW..',  # 4
     '...BW...',  # 5
     '...BB...',  # 6
     '...B....',  # 7
     '........']  # 8
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    e3_board = from_array_to_dictionary_board(e3_board)
    f3_board = from_array_to_dictionary_board(f3_board)
    g4_board = from_array_to_dictionary_board(g4_board)
    f5_board = from_array_to_dictionary_board(f5_board)
    c6_board = from_array_to_dictionary_board(c6_board)
    f6_board = from_array_to_dictionary_board(f6_board)
    d7_board = from_array_to_dictionary_board(d7_board)
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
game_state_e3 = TwoPlayerGameState(
    game=game,
    board=e3_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_f3 = TwoPlayerGameState(
    game=game,
    board=f3_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_g4 = TwoPlayerGameState(
    game=game,
    board=g4_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_f5 = TwoPlayerGameState(
    game=game,
    board=f5_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_c6 = TwoPlayerGameState(
    game=game,
    board=c6_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_f6 = TwoPlayerGameState(
    game=game,
    board=f6_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_d7 = TwoPlayerGameState(
    game=game,
    board=d7_board,
    initial_player=player_b,
    player_max=player_a,
)

print("Given this initial state:")
print('........\n',  # 1
      '........\n',  # 2
      '...B....\n',  # 3
      '...BWW..\n',  # 4
      '...BW...\n',  # 5
      '...WB...\n',  # 6
      '........\n',  # 7
      '........\n',  # 8
      )
print("This is the ranking of the possible movements: E3, F3, G4, F5, C6, F6, D7")
print("1- E3  -- your score for this movement:", heuristic_1(game_state_e3),
      " <- This is the best board (so, your strategy should prefer it).")
print("2- F3  -- your score for this movement:", heuristic_1(game_state_f3))
print("3- G4  -- your score for this movement:", heuristic_1(game_state_g4))
print("4- F5  -- your score for this movement:",
      heuristic_1(game_state_f5), " <- Same score as C6.")
print("4- C6  -- your score for this movement:",
      heuristic_1(game_state_c6), " <- Same score as F5.")
print("6- F6  -- your score for this movement:", heuristic_1(game_state_f6))
print("7- D7  -- your score for this movement:", heuristic_1(game_state_d7),
      " <- This is the worst board (so, your strategy should not prefer it).")
