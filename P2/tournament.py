"""Infrastructure for tournament.

   Author:
        Alejandro Bellogin <alejandro.bellogin@uam.es>
"""

from __future__ import annotations  # For Python 3.7

import inspect  # for dynamic members of a module
import os
import sys
import time
from abc import ABC
from importlib import util
import traceback
from typing import Callable, Tuple

from game import Player, TwoPlayerGameState, TwoPlayerMatch
from heuristic import Heuristic
from strategy import MinimaxStrategy
"""
NOTE: When MinimaxAlphaBetaStrategy has been implemented
replace MinimaxAlphaBetaStrategy for MinimaxStrategy,
so that the tournament runs faster.
"""
# from strategy import MinimaxAlphaBetaStrategy


class StudentHeuristic(ABC):
    def __init__(self):
        pass

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        pass

    def get_name(self) -> str:
        pass


class Tournament(object):
    def __init__(self, max_depth: int,
                 init_match: Callable[[Player, Player], TwoPlayerMatch],
                 max_evaluation_time: float):
        self.__max_depth = max_depth
        self.__init_match = init_match
        self.__max_eval_time = max_evaluation_time

    def __get_function_from_str(self, name: str, definition: str, max_strat: int) -> list:
        # write content in file with new name
        newfile = "playermodule__" + name
        with open(newfile, 'w') as fp:
            print(definition, file=fp)
        student_classes = list()
        n_strat = 0
        # not needed, but this hack somehow fixes some files not being loaded
        time.sleep(1)
        sp = util.find_spec(newfile.replace(".py", ""))
        if sp:
            m = sp.loader.load_module()
            # return all the objects that satisfy the function signature
            for name, obj in inspect.getmembers(m, inspect.isclass):
                if name != "StudentHeuristic":
                    for name2, obj2 in inspect.getmembers(obj, inspect.isfunction):
                        if name2 == "evaluation_function" and n_strat < max_strat:
                            student_classes.append(obj)
                            n_strat += 1
                        elif name2 == "evaluation_function":
                            print("Ignoring evaluation function in %s because limit of submissions was reached (%d)" % (
                                name, max_strat), file=sys.stderr)
                    # end for
            # end for
        # remove file
        os.remove(newfile)
        return student_classes

    #   we assume there is one file for each student/pair
    def load_strategies_from_folder(self, folder: str, max_strat: int = 3) -> dict:
        student_strategies = dict()
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                with open(p, 'r') as fp:
                    s = fp.read()
                    name = f
                    strategies = self.__get_function_from_str(
                        name, s, max_strat)
                    student_strategies[f] = strategies
        return student_strategies

    def run(self, student_strategies: dict, increasing_depth: bool = True,
            n_pairs: int = 1, allow_selfmatch: bool = False) -> Tuple[dict, dict, dict]:
        """
        Play a tournament among the strategies.
        n_pairs = games each strategy plays as each color against
        each opponent. So with N strategies, a total of
        N*(N-1)*n_pairs games are played.
        """
        scores = dict()
        totals = dict()
        name_mapping = dict()
        for student1 in student_strategies:
            strats1 = student_strategies[student1]
            for student2 in student_strategies:
                if student1 > student2:
                    continue
                if student1 == student2 and not allow_selfmatch:
                    continue
                strats2 = student_strategies[student2]
                for player1 in strats1:
                    for player2 in strats2:
                        # we now instantiate the players
                        for pair in range(2*n_pairs):
                            player1_first = (pair % 2) == 1
                            sh1 = player1()
                            name1 = student1 + "_" + sh1.get_name()
                            name_mapping[name1] = sh1.get_name()
                            sh2 = player2()
                            name2 = student2 + "_" + sh2.get_name()
                            name_mapping[name2] = sh2.get_name()
                            if increasing_depth:
                                for depth in range(1, self.__max_depth):
                                    pl1 = Player(
                                        name=name1,
                                        strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
                                            heuristic=Heuristic(
                                                name=sh1.get_name(),
                                                evaluation_function=sh1.evaluation_function),
                                            max_depth_minimax=depth,
                                            max_sec_per_evaluation=self.__max_eval_time,
                                            verbose=0,
                                        ),
                                    )
                                    pl2 = Player(
                                        name=name2,
                                        strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
                                            heuristic=Heuristic(
                                                name=sh2.get_name(),
                                                evaluation_function=sh2.evaluation_function),
                                            max_depth_minimax=depth,
                                            max_sec_per_evaluation=self.__max_eval_time,
                                            verbose=0,
                                        ),
                                    )

                                    self.__single_run(
                                        player1_first, pl1, name1, pl2, name2, scores, totals)
                            else:
                                depth = self.__max_depth
                                pl1 = Player(
                                    name=name1,
                                    strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
                                        heuristic=Heuristic(
                                            name=sh1.get_name(),
                                            evaluation_function=sh1.evaluation_function),
                                        max_depth_minimax=depth,
                                        max_sec_per_evaluation=self.__max_eval_time,
                                        verbose=0,
                                    ),
                                )
                                pl2 = Player(
                                    name=name2,
                                    strategy=MinimaxStrategy(  # MinimaxAlphaBetaStrategy(
                                        heuristic=Heuristic(
                                            name=sh2.get_name(),
                                            evaluation_function=sh2.evaluation_function),
                                        max_depth_minimax=depth,
                                        max_sec_per_evaluation=self.__max_eval_time,
                                        verbose=0,
                                    ),
                                )

                                self.__single_run(
                                    player1_first,
                                    pl1, name1,
                                    pl2, name2,
                                    scores, totals)
        return scores, totals, name_mapping

    def __single_run(self, player1_first: bool, pl1: Player, name1: str,
                     pl2: Player, name2: str, scores: dict, totals: dict):
        players = []
        if player1_first:
            players = [pl1, pl2]
        else:
            players = [pl2, pl1]
        game = self.__init_match(players[0], players[1])
        try:
            game_scores = game.play_match()
            # let's get the scores (do not assume they will always be binary)
            # we assume a higher score is better
            if player1_first:
                score1, score2 = game_scores[0], game_scores[1]
            else:
                score1, score2 = game_scores[1], game_scores[0]
            wins = loses = 0
            if score1 > score2:
                wins, loses = 1, 0
            elif score2 > score1:
                wins, loses = 0, 1
        except Warning:
            wins = loses = 0
        # store the 1-to-1 numbers
        if name1 not in scores:
            scores[name1] = dict()
        if name2 not in scores:
            scores[name2] = dict()
        scores[name1][name2] = wins if name2 not in scores[name1] else wins + \
            scores[name1][name2]
        scores[name2][name1] = loses if name1 not in scores[name2] else loses + \
            scores[name2][name1]
        # store the total values
        if name1 not in totals:
            totals[name1] = 0
        totals[name1] += wins
        if name2 not in totals:
            totals[name2] = 0
        totals[name2] += loses
        # end of function

    def test_reversi_strategies_with_puzzles(self, strategies: dict,
                                             map_name_boardstate,
                                             map_name_initialboard,
                                             gamestate_fun) -> Tuple[dict, dict]:
        test_scores = dict()
        test_scores_plain = dict()
        # inline import to keep tournament generic
        from reversi import from_array_to_dictionary_board
        #
        for name in strategies:
            strategy_results = dict()
            test_scores[name] = strategy_results
            strats = strategies[name]
            for player in strats:
                try:
                    player_results = dict()
                    player_obj = player()
                    # same as in tournament.run_symmetric
                    normalised_name = name + "_" + \
                        player_obj.__class__.__name__ + "__" + player_obj.get_name()
                    test_scores_plain[normalised_name] = player_results
                    strategy_results[player_obj.get_name()] = player_results
                    for n, boardstate in map_name_boardstate.items():
                        inistate = map_name_initialboard[n[0]]
                        state = gamestate_fun(
                            inistate, boardstate, player_obj, self.__max_depth, self.__max_eval_time)
                        score = player_obj.evaluation_function(state)
                        player_results[n] = score
                    # another score based on state.move()
                    for n0, inistate in map_name_initialboard.items():
                        # get state from initial board
                        state = gamestate_fun(
                            inistate, inistate, player_obj, self.__max_depth, self.__max_eval_time)
                        next_state = state.move()
                        next_board = next_state.board
                        # compare value with possibilities, get ranking/score from groundtruth
                        found_board = ""
                        for n, boardstate in map_name_boardstate.items():
                            if n[0] == n0:
                                boardstate_dict = from_array_to_dictionary_board(boardstate)
                                if next_board == boardstate_dict:
                                    found_board = n[1]
                                    break
                        player_results[(n0, "initial_board")] = found_board
                except BaseException as error:  # let's catch anything else
                    print('Not able to test player {}: {}'.format(
                        player, error), file=sys.stderr)
                    print(str(player), file=sys.stderr)
                    print("".join(traceback.TracebackException.from_exception(
                        error).format()), file=sys.stderr)
                    continue
        return test_scores, test_scores_plain
