"""Test case for an heuristic.

Authors:
    Alejandro Bellogin <alejandro.bellogin@uam.es>
    Daniel Fernandez <daniel.fernandezs@uam.es>

"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from p2_1322_11_Piñón_Rincón import CornerHeuristic, MobilityHeuristic, PieceAdvantageHeuristic
from game import Player, TwoPlayerGameState
from heuristic import Heuristic
from reversi import (
    Reversi,
    from_array_to_dictionary_board,
)
from strategy import (
    MinimaxStrategy,
)

corner_heuristic_instance = CornerHeuristic()
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
        heuristic=Heuristic(name='corner_control',
                            evaluation_function=CornerHeuristic().evaluation_function),
        max_depth_minimax=3,
        verbose=0,
    ),
)

player_minimax_2 = Player(
    name='White',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='corner_control_white',
                            evaluation_function=CornerHeuristic().evaluation_function),
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
    ['........',  # 1
     '........',  # 2
     '..BW.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '..WWBBWB',  # 6
     '..WWWWWW',  # 7
     '.WWW.WB.']  # 8
)

h8_board = (
    ['........',  # 1
     '........',  # 2
     '..BW.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '..WWBBWB',  # 6
     '..WWWWBB',  # 7
     '.WWW.WBB']  # 8
)

b6_board = (
    ['........',  # 1
     '........',  # 2
     '..BW.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '.BBBBBWB',  # 6
     '..WWWWWW',  # 7
     '.WWW.WB.']  # 8
)

d2_board = (
    ['........',  # 1
     '...B....',  # 2
     '..BB.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '..WWBBWB',  # 6
     '..WWWWWW',  # 7
     '.WWW.WB.']  # 8
)

c2_board = (
    ['........',  # 1
     '..B.....',  # 2
     '..BB.BBB',  # 3
     '...BBWBB',  # 4
     '.BBBWBWB',  # 5
     '..WWBBWB',  # 6
     '..WWWWWW',  # 7
     '.WWW.WB.']  # 8
)

e3_board = (
    ['........',  # 1
     '........',  # 2
     '..BBBBBB',  # 3
     '...BBBBB',  # 4
     '.BBBBBBB',  # 5
     '..WWBBWB',  # 6
     '..WWWWWW',  # 7
     '.WWW.WB.']  # 8
)

b7_board = (
    ['........',  # 1
     '........',  # 2
     '..BW.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '..BWBBWB',  # 6
     '.BWWWWWW',  # 7
     '.WWW.WB.']  # 8
)

e8_board = (
    ['........',  # 1
     '........',  # 2
     '..BW.BBB',  # 3
     '...BWWBB',  # 4
     '.BBBWBWB',  # 5
     '..BWBBBB',  # 6
     '..WBBBWW',  # 7
     '.WWWBBB.']  # 8
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    h8_board = from_array_to_dictionary_board(h8_board)
    b6_board = from_array_to_dictionary_board(b6_board)
    d2_board = from_array_to_dictionary_board(d2_board)
    c2_board = from_array_to_dictionary_board(c2_board)
    e3_board = from_array_to_dictionary_board(e3_board)
    b7_board = from_array_to_dictionary_board(b7_board)
    e8_board = from_array_to_dictionary_board(e8_board)
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
game_state_h8 = TwoPlayerGameState(
    game=game,
    board=h8_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_b6 = TwoPlayerGameState(
    game=game,
    board=b6_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_d2 = TwoPlayerGameState(
    game=game,
    board=d2_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_c2 = TwoPlayerGameState(
    game=game,
    board=c2_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_e3 = TwoPlayerGameState(
    game=game,
    board=e3_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_b7 = TwoPlayerGameState(
    game=game,
    board=b7_board,
    initial_player=player_b,
    player_max=player_a,
)

# Initialize a game state.
game_state_e8 = TwoPlayerGameState(
    game=game,
    board=e8_board,
    initial_player=player_b,
    player_max=player_a,
)

print("Given this initial state:")
print(' ........\n',  # 1
      '........\n',  # 2
      '..BW.BBB\n',  # 3
      '...BWWBB\n',  # 4
      '.BBBWBWB\n',  # 5
      '..WWBBWB\n',  # 6
      '..WWWWWW\n',  # 7
      '.WWW.WB.\n'  # 8
      )
print("This is the ranking of the possible movements: H8, B6, D2, C2, E3, B7, E8.")
print("1- H8  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_h8),
      " <- This is the best board (so, your strategy should prefer it).")
print("2- B6  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_b6))
print("3- D2  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_d2))
print("4- C2  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_c2))
print("5- E3  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_e3))
print("6- B7  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_b7))
print("7- E8  -- your score for this movement:", corner_heuristic_instance.evaluation_function(game_state_e8),
      " <- This is the worst board (so, your strategy should not prefer it).")
