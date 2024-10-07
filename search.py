# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Name student 1: Noelia Rincón Roldán
Name student 2: Ernesto Piñón Esteban
IA lab group and pair: 1322 - 11

"""

import util
from game import Directions


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(search_problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(search_problem):
    """"
    Search the deepest nodes in the search tree first (DFS).

    This function should return a list of actions to reach the goal.
    """
    return generic(search_problem, util.Stack())

def generic(search_problem, structure,priority_queue):

    start_state = search_problem.getStartState()
    
    if priority_queue:
        structure.push((start_state, [], 0))
    else:
        structure.push((start_state, [], 0),0)  
    visited = []

    while not structure.isEmpty():
        current_state, directions, current_cost = structure.pop()

        if search_problem.isGoalState(current_state):
            return directions

        if current_state not in visited:
            visited.append(current_state)

            for successor, direction, cost in search_problem.getSuccessors(current_state):
                if successor not in visited:
                    new_directions = directions + [direction]
                    if priority_queue:  
                        structure.push((successor, new_directions, current_cost+cost),current_cost + cost)
                    else:                      
                        structure.push((successor, new_directions, current_cost + cost))

    return None




def breadthFirstSearch(search_problem):
    """Search the shallowest nodes in the search tree first."""  
    return generic(search_problem, util.Queue())


def uniformCostSearch(search_problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    structure = util.PriorityQueue()
    start_state = search_problem.getStartState()
    structure.push((start_state, [], 0),0)  
    visited = []


    while not structure.isEmpty():
        
        current_state, directions, current_cost = structure.pop()

        
        if search_problem.isGoalState(current_state):
            return directions  

        
        if current_state not in visited:
            visited.append(current_state)  

            
            for successor, direction, cost in search_problem.getSuccessors(current_state):
                if successor not in visited:
                    
                    new_directions = directions + [direction]
                    
                    structure.push((successor, new_directions, current_cost+cost),current_cost + cost)

    return None



def nullHeuristic(state, search_problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(search_problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    structure = util.PriorityQueue()
    start_state = search_problem.getStartState()
    structure.push((start_state, [], 0), 0)

    visited = []

    while not structure.isEmpty():
        current_state, directions, current_cost = structure.pop() 
    
        if search_problem.isGoalState(current_state):
            return directions
        
        if current_state not in visited:
            visited.append(current_state)

            for successor, direction, cost in search_problem.getSuccessors(current_state):
                if successor not in visited:

                    priority = heuristic(successor, search_problem) + (current_cost + cost) 
                    structure.push((successor, directions + [direction], current_cost + cost), priority) 

    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
