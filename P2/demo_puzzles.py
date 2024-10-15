"""Allows to tests several puzzles from a file.

   Authors:
        Alejandro Bellogin <alejandro.bellogin@uam.es>
        Daniel Fernandez <daniel.fernandezs@uam.es>
"""

from __future__ import annotations  # For Python 3.7

from typing import Sequence
import numpy as np
import json

from tabulate import tabulate  # Para imprimir la tabla de forma legible

from game import Player, TwoPlayerGameState, TwoPlayerMatch
from heuristic import Heuristic
from reversi import (
    Reversi,
    from_array_to_dictionary_board,
)
from tournament import (
    Tournament,
    StudentHeuristic,
)
from strategy import MinimaxStrategy
"""
NOTE: When MinimaxAlphaBetaStrategy has been implemented
replace MinimaxAlphaBetaStrategy for MinimaxStrategy,
so that the tournament runs faster.
"""
# from strategy import MinimaxAlphaBetaStrategy


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


class SubtractionHeuristic(StudentHeuristic):
    def get_name(self) -> str:
        return "subtraction_heuristic"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        return subtraction_heuristic(state)


class Solution1(StudentHeuristic):
    def get_name(self) -> str:
        return "solution1"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        # let's use an auxiliary function
        aux = self.dummy(123)
        return aux

    def dummy(self, n: int) -> int:
        return n + 1


def create_state(prev_board, initial_board, strategy, depth, max_time) -> TwoPlayerGameState:
    player1 = Player(
        name="player1",
        strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
            heuristic=Heuristic(name=strategy.get_name(
            ), evaluation_function=strategy.evaluation_function),
            max_depth_minimax=depth,
            max_sec_per_evaluation=max_time,
            verbose=0,
        ),
    )
    player2 = Player(
        name="player2",
        strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
            heuristic=Heuristic(name=strategy.get_name(
            ), evaluation_function=strategy.evaluation_function),
            max_depth_minimax=depth,
            max_sec_per_evaluation=max_time,
            verbose=0,
        ),
    )
    height = len(initial_board)
    width = len(initial_board[0])
    try:
        initial_board = from_array_to_dictionary_board(initial_board)
        previous_board = from_array_to_dictionary_board(prev_board)
    except ValueError:
        raise ValueError('Wrong configuration of the board')
    else:
        # print("Successfully initialised board from array")
        pass

    # Initialize a reversi game.

    game = Reversi(
        player1=player1,  # needed for "mov" metric
        player2=player2,  # useful for heuristic
        height=height,
        width=width,
    )
    prev_game_state = TwoPlayerGameState(
        game=game,
        board=previous_board,
        initial_player=player2,
        player_max=player2,
    )
    game_state = TwoPlayerGameState(
        game=game,
        board=initial_board,
        initial_player=player1,  
        player_max=player1,    
        parent=prev_game_state,
    )
    game_state.end_of_game, game_state.scores = game_state.game.score(
        game_state)
    return game_state


def create_match(player1: Player, player2: Player, initial_board, n_moves, max_sec) -> TwoPlayerMatch:
    height = len(initial_board)
    width = len(initial_board[0])
    try:
        initial_board = from_array_to_dictionary_board(initial_board)
    except ValueError:
        raise ValueError('Wrong configuration of the board')
    else:
        # print("Successfully initialised board from array")
        pass

    game = Reversi(
        player1=player1,
        player2=player2,
        height=height,
        width=width,
    )
    initial_player = player1
    game_state = TwoPlayerGameState(
        game=game,
        board=initial_board,
        initial_player=initial_player,
    )
    match = TwoPlayerMatch(game_state, n_moves_max=n_moves,
                           max_sec_per_move=max_sec, gui=False,)
    return match


def compute_puzzle_metrics(ranking, real_ranking):
    metric_dict = dict()
    # we assume only top-1 puzzle is valid
    groundtruth = [x[0] for x in real_ranking if x[1] <= 1]
    # p@1
    ret = [x[0] for x in ranking[:1]]
    metric_dict["p@1"] = sum([1 if x in groundtruth else 0 for x in ret]) / 1.0
    # p@2
    ret = [x[0] for x in ranking[:2]]
    metric_dict["p@2"] = sum([1 if x in groundtruth else 0 for x in ret]) / 2.0
    # now we assume top-2 are valid
    groundtruth = [x[0] for x in real_ranking if x[1] <= 2]
    # pp@1
    ret = [x[0] for x in ranking[:1]]
    metric_dict["pp@1"] = sum([1 if x in groundtruth else 0 for x in ret]) / 1.0
    # pp@2
    ret = [x[0] for x in ranking[:2]]
    metric_dict["pp@2"] = sum([1 if x in groundtruth else 0 for x in ret]) / 2.0
    # global value (based on nDCG)
    
    from math import log2
    dcg = 0.0
    idcg = 0.0
    rank = 1
    for b, _ in ranking:

        # find the true ranking
        for tb, tr in real_ranking:
            if tb == b:
                # relevance = 1.0/tr  # 1, 0.5, 0.33, 0.25, ... # XXX
                relevance = len(real_ranking) - tr  # n, n-1, n-2, ...
                idcg += (pow(2, relevance) - 1) / log2(tr + 1)
                dcg += (pow(2, relevance) - 1) / log2(rank + 1)
                break

        # update ranking
        rank = rank + 1
    metric_dict["global"] = dcg / idcg if idcg != 0 else 0
    # print(metric_dict["global"])
    # print("\t", ranking)
    # print("\t", real_ranking)
    return metric_dict


def get_empty_board():
    initial_board = (
        ['........',
         '........',
         '........',
         '...WB...',
         '...BW...',
         '........',
         '........',
         '........']
    )
    return initial_board


filename_with_puzzles = "puzzles.json"


def match_builder(pl1, pl2): return create_match(
    pl1, pl2, get_empty_board(), 100, 100)


tour = Tournament(max_depth=3, init_match=match_builder,
                  max_evaluation_time=0.5)
# add here all the strategies (as subclass of StudentHeuristic) that want to be tested
strats = {"sh_strategy": [SubtractionHeuristic, Solution1]}

with open(filename_with_puzzles) as json_file:
    map_from_file = json.load(json_file)
    # let's flat the states to make the request only once
    all_boardstates = dict()
    all_initialboardstates = dict()
    # and transform data into a dict for easier access
    puzzle_dict = dict()

    for puzzle in map_from_file:
        if puzzle["puzzle_game"] != "reversi":
            continue

        name = puzzle["puzzle_name"]
        puzzle_dict[name] = dict()
        puzzle_dict[name]["ranking"] = list()

        for b in puzzle["ranking"]:
            puzzle_dict[name]["ranking"].append((b, puzzle["ranking"][b]))

        # store the ranking sorted by (ascending) rank
        puzzle_dict[name]["ranking"] = sorted(
            puzzle_dict[name]["ranking"], key=lambda x: x[1], reverse=False
        )

        puzzle_dict[name]["boards"] = list()

        for board in puzzle["boards"]:
            bname = board["name"]
            barray = board["board_array"]

            if bname == "initial_board":
                # let's avoid testing the initial board
                # but use it to create previous state
                all_initialboardstates[name] = barray

            else:
                all_boardstates[(name, bname)] = barray
                puzzle_dict[name]["boards"].append(bname)

    test_scores, test_scores_plain = tour.test_reversi_strategies_with_puzzles(strategies=strats,
                                                                               map_name_boardstate=all_boardstates,
                                                                               map_name_initialboard=all_initialboardstates,
                                                                               gamestate_fun=create_state)

    # XXX
    # compute results
    results = dict()
    for strategy_name in test_scores:
        strategy_results = dict()
        results[strategy_name] = strategy_results
        for player_name in test_scores[strategy_name]:
            # collect all the scores produced by this player
            player_scores = dict()
            for state_name in test_scores[strategy_name][player_name]:
                puzzle_name, board_name = state_name
                if puzzle_name not in player_scores:
                    player_scores[puzzle_name] = dict()
                player_scores[puzzle_name][board_name] = test_scores[strategy_name][player_name][state_name]
            # end for
            # compute all the results of this player
            player_results = dict()
            strategy_results[player_name] = player_results
            # sort results per puzzle and compare against puzzle ranking
            for puzzle_name in player_scores:
                # sorted_boards = sorted(player_scores[puzzle_name], key=lambda x: (x[1], x[0]), reverse=True)
                # sorted boards may not be sorted if the heuristic is trivial
                # hence, we will add the boards in the list in the reverse order wrt groundtruth
                # so that, if sorting does nothing, it will not benefit those trivial heuristics
                groundtruth = puzzle_dict[puzzle_name]["ranking"]
                sorted_boards = list()
                for bname, _ in groundtruth[::-1]:
                    sorted_boards.append(
                        (bname, player_scores[puzzle_name][bname]))
                sorted_boards = sorted(
                    sorted_boards, key=lambda x: x[1], reverse=True)
                player_results[puzzle_name] = compute_puzzle_metrics(
                    sorted_boards, groundtruth)
                # compute new score based on running move
                found_board = player_scores[puzzle_name]["initial_board"]
                found_ranking = -1
                for name, pos in groundtruth:
                    if found_board == name:
                        found_ranking = pos
                        break
                player_results[puzzle_name]["mov"] = found_ranking
    print(results)

    results_plain = dict()
    for strategy_name in test_scores_plain:
        
        player_results = dict()
        # collect all the scores produced by this player
        player_scores = dict()

        results_plain[strategy_name] = player_results
        for state_name in test_scores_plain[strategy_name]:
            puzzle_name, board_name = state_name
        
            if puzzle_name not in player_scores:
                player_scores[puzzle_name] = dict()
            player_scores[puzzle_name][board_name] = test_scores_plain[strategy_name][state_name]
        # end for
        # compute all the results of this player
        # sort results per puzzle and compare against puzzle ranking

        for puzzle_name in player_scores:
            # sorted_boards = sorted(player_scores[puzzle_name], key=lambda x: (x[1], x[0]), reverse=True)
            # sorted boards may not be sorted if the heuristic is trivial
            # hence, we will add the boards in the list in the reverse order wrt groundtruth
            # so that, if sorting does nothing, it will not benefit those trivial heuristics
            groundtruth = puzzle_dict[puzzle_name]["ranking"]
            sorted_boards = list()
            for bname, _ in groundtruth[::-1]:
                sorted_boards.append(
                    (bname, player_scores[puzzle_name][bname]))
            sorted_boards = sorted(
                sorted_boards, key=lambda x: x[1], reverse=True)
            player_results[puzzle_name] = compute_puzzle_metrics(
                sorted_boards, groundtruth
            )

            # compute new score based on running move
            found_board = player_scores[puzzle_name]["initial_board"]
            found_ranking = -1
            for name, pos in groundtruth:
                if found_board == name:
                    found_ranking = pos
                    break
            player_results[puzzle_name]["mov"] = found_ranking
    print(results_plain)

    #   compute the average
    average_results = dict()
    for normalised_name in results_plain:
        player_results = results_plain[normalised_name]
        avg_result1 = 0.0
        avg_result2 = 0.0
        avg_result3 = 0.0
        avg_result4 = 0.0
        avg_count = 0
        for puzzle in player_results:
            # if any other metric should be considered, change here
            avg_result1 += player_results[puzzle]["p@1"]
            avg_result2 += player_results[puzzle]["pp@1"]
            avg_result3 += player_results[puzzle]["global"]
            avg_result4 += player_results[puzzle]["mov"]
            avg_count += 1
        avg_result1 = avg_result1 / avg_count if avg_count > 0 else 0.0
        avg_result2 = avg_result2 / avg_count if avg_count > 0 else 0.0
        avg_result3 = avg_result3 / avg_count if avg_count > 0 else 0.0
        avg_result4 = avg_result4 / avg_count if avg_count > 0 else 0.0
        average_results[normalised_name] = (
            # avg_result1, avg_result2, avg_result3, avg_result4)
            round(avg_result1, 3), round(avg_result2, 3), round(avg_result3, 3), round(avg_result4, 3)
        )

    # Imprimir los resultados en formato de tabla
    headers = ["Strategy", "P@1", "PP@1", "Global", "Mov"]
    table_data = [(name, *metrics) for name, metrics in average_results.items()]
    print(tabulate(table_data, headers=headers, floatfmt=".3f"))

