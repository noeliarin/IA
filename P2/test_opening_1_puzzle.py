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
from p2_1322_11_Piñón_Rincón import ErnesyNoelia

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


# Jugador usando ErnesyNoelia
player_minimax_1 = Player(
    name='Black',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='corner_control',
                            evaluation_function=ErnesyNoelia().evaluation_function),
        max_depth_minimax=3,
        verbose=0,
    ),
)

player_minimax_2 = Player(
    name='White',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='corner_control_white',
                            evaluation_function=ErnesyNoelia().evaluation_function),
        max_depth_minimax=3,
        verbose=0,
    ),
)


# minimax vs minimax player
player_a, player_b = player_minimax_1, player_minimax_2

"""
Diagonal opening test
"""
initial_board = (
    ['........',  # 1
     '........',  # 2
     '........',  # 3
     '...WB...',  # 4
     '...BWB..',  # 5
     '.....W..',  # 6
     '........',  # 7
     '........']  # 8
)

c4_board = (
    ['........',  # 1
     '........',  # 2
     '........',  # 3
     '..BBB...',  # 4
     '...BWB..',  # 5
     '.....W..',  # 6
     '........',  # 7
     '........']  # 8
)

d3_board = (
    ['........',  # 1
     '........',  # 2
     '...B....',  # 3
     '...BB...',  # 4
     '...BWB..',  # 5
     '.....W..',  # 6
     '........',  # 7
     '........']  # 8
)

e6_board = (  # This is the best option between c4, d3 and f7
    ['........',  # 1
     '........',  # 2
     '........',  # 3
     '...WB...',  # 4
     '...BBB..',  # 5
     '....BW..',  # 6
     '........',  # 7
     '........']  # 8
)

f7_board = (
    ['........',  # 1
     '........',  # 2
     '........',  # 3
     '...WB...',  # 4
     '...BWB..',  # 5
     '.....B..',  # 6
     '.....B..',  # 7
     '........']  # 8
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    c4_board = from_array_to_dictionary_board(c4_board)
    d3_board = from_array_to_dictionary_board(d3_board)
    e6_board = from_array_to_dictionary_board(e6_board)
    f7_board = from_array_to_dictionary_board(f7_board)
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
game_state_c4 = TwoPlayerGameState(
    game=game,
    board=c4_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_d3 = TwoPlayerGameState(
    game=game,
    board=d3_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_e6 = TwoPlayerGameState(
    game=game,
    board=e6_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_f7 = TwoPlayerGameState(
    game=game,
    board=f7_board,
    initial_player=player_b,
    player_max=player_a,
)

print("Given this initial state:")
print(' ........\n',  # 1
      '........\n',  # 2
      '........\n',  # 3
      '...WB...\n',  # 4
      '...BWB..\n',  # 5
      '.....W..\n',  # 6
      '........\n',  # 7
      '........\n'  # 8
      )
print("This is the ranking of the possible movements: E6, D3, C4, F7")
print("1- E6  -- your score for this movement:", ErnesyNoelia().evaluation_function(game_state_e6))
print("2- D3  -- your score for this movement:", ErnesyNoelia().evaluation_function(game_state_d3))
print("3- C4  -- your score for this movement:", ErnesyNoelia().evaluation_function(game_state_c4))
print("4- F7  -- your score for this movement:", ErnesyNoelia().evaluation_function(game_state_f7))