from game import TwoPlayerGameState
from tournament import StudentHeuristic
import random

class ErnesyNoelia(StudentHeuristic):
    def get_name(self) -> str:
        return "Heuristica de Noe y Ernes"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player = state.next_player
        opponent = state.game.opponent(player)
        board = state.board

        # Valor de las casillas estratégicas (esquinas, bordes)
        strategic_map = [
            [100, -25, 0, 0, 0, 0, -25, 100],  
            [-25, -50, 0, 0, 0, 0, -50, -25],  
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 10, 10, 0, 0, 0],        
            [0, 0, 0, 10, 10, 0, 0, 0],        
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [-25, -50, 0, 0, 0, 0, -50, -25],  
            [100, -25, 0, 0, 0, 0, -25, 100]  
        ]
        
        player_score = 0
        opponent_score = 0

        for i in range(8):
            for j in range(8):
                cell_value = strategic_map[i][j]
                if board.get((i, j)) == player.label:
                    player_score += cell_value
                elif board.get((i, j)) == opponent.label:
                    opponent_score += cell_value

        player_moves = len(state.game.generate_successors(state))
        opponent_moves = len(state.game.generate_successors(state)) 
        
        if opponent_moves == 0:
            mobility_score = 100
        else:
            mobility_score = -10 * opponent_moves

        total_score = player_score - opponent_score + mobility_score

        random_factor = random.uniform(-5, 5)
        total_score += random_factor

        return total_score


