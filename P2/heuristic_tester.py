"""Test cases for heuristics.

Authors:
    Alejandro Bellogin <alejandro.bellogin@uam.es>

"""

from __future__ import annotations  # For Python 3.7

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


def eval_function1(state: TwoPlayerGameState) -> float:
    """Return a random value, except for terminal game states."""
    state_value = 2*np.random.rand() - 1
    if state.end_of_game:
        scores = state.scores
    else:
        # print(state.next_player.label)
        # print(state.game.opponent(state.next_player).label)
        scores = [
            state.game._player_coins(state.board, state.next_player.label),
            state.game._player_coins(state.board, state.game.opponent(state.next_player).label)
            ]
        # print(scores)

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


def eval_function2(state: TwoPlayerGameState) -> float:
    """Return a random value, except for terminal game states."""
    state_value = 2*np.random.rand() - 1
    if state.end_of_game:
        scores = state.scores
    else:
        # print(state.next_player.label)
        # print(state.game.opponent(state.next_player).label)
        scores = [
            state.game._player_coins(state.board, state.next_player.label),
            state.game._player_coins(state.board, state.game.opponent(state.next_player).label)]
        # print(scores)

    # Evaluation of the state from the point of view of MAX
    assert isinstance(scores, (Sequence, np.ndarray))
    score_difference = scores[0] - scores[1]

    if state.is_player_max(state.player1):
        state_value = score_difference
    elif state.is_player_max(state.player2):
        state_value = - score_difference
    else:
        raise ValueError('Player MAX not defined')

    # BUT now, the function is actually returning the opposite value, 
    # so the point of view is that from MIN
    return - state_value


player_minimax_1 = Player(
    name='direct',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='eval1', evaluation_function=eval_function1),
        max_depth_minimax=3,
        verbose=0,
    ),
)

player_minimax_2 = Player(
    name='inverse',
    strategy=MinimaxStrategy(
        heuristic=Heuristic(name='eval2', evaluation_function=eval_function2),
        max_depth_minimax=3,
        verbose=0,
    ),
)


# minimax vs minimax player
player_a, player_b = player_minimax_1, player_minimax_2

"""
Here you can initialize the player that moves first
and the board to any valid state.
E.g., it can be an intermediate state.
"""
# Board at an intermediate state of the game.
initial_board = (
    ['..B.B..',
     '.WBBW..',
     'WBWBB..',
     '.W.WWW.',
     '.BBWBWB']
)

initial_board2 = (
    ['..B.BBB',
     '.WBBW..',
     'WBWBB..',
     '.W.WWW.',
     '.BBWBWB']
)

height = len(initial_board)
width = len(initial_board[0])
try:
    initial_board = from_array_to_dictionary_board(initial_board)
    initial_board2 = from_array_to_dictionary_board(initial_board2)
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
game_state = TwoPlayerGameState(
    game=game,
    board=initial_board,
    initial_player=player_a,
    player_max=player_a,
)

game_state2 = TwoPlayerGameState(
    game=game,
    board=initial_board,
    initial_player=player_a,
    player_max=player_b,
)

game_state3 = TwoPlayerGameState(
    game=game,
    board=initial_board2,
    initial_player=player_a,
    player_max=player_a,
)

# CASE 1: check how two functions may return different values for the same state if the heuristic is not properly computed
print("Case 1")
print("\t", eval_function1(game_state))
print("\t", eval_function2(game_state))
#       output should be higher when heuristic considers the point of view of MAX (eval_function1)
# This test can also be used only with one function to check that your function returns positive values when MAX is playing

# CASE 2: check how the same function returns different values depending on who is max
print("Case 2")
print("\t", eval_function1(game_state))
print("\t", eval_function1(game_state2))
#       output should be higher for the situation where max is player a
#       (since this simple heuristic only checks number of coins)

# CASE 3: check how the same function returns different values depending on the board status
print("Case 3")
print("\t", eval_function1(game_state))
print("\t", eval_function1(game_state3))
#       output should be higher for the situation where MAX player has a better position
#       since this simple heuristic only checks number of coins, game_state3 is better
