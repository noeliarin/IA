from game import TwoPlayerGameState
from tournament import StudentHeuristic

class CornerHeuristic(StudentHeuristic):
    def get_name(self) -> str:
        return "corner_control_dynamic_v2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player = state.next_player
        opponent = state.game.opponent(player)
        board = state.board

        # Mapa de valores de las posiciones del tablero
        value_map = [
            [100, -10, 10, 5, 5, 10, -10, 100],  # Fila 0
            [-10, -20, 1, 1, 1, 1, -20, -10],   # Fila 1
            [10, 1, 1, 1, 1, 1, 1, -10],        # Fila 2
            [5, 1, 1, 1, 1, 1, 1, 5],          # Fila 3
            [5, 1, 1, 1, 1, 1, 1, 5],          # Fila 4
            [10, 1, 1, 1, 1, 1, 1, -10],       # Fila 5
            [-10, -20, 1, 1, 1, 1, -20, -10],   # Fila 6
            [100, -10, 10, 5, 5, 10, -10, 100]   # Fila 7
        ]

        score = 0  # Puntaje total

        # Calcular la puntuación según el mapa de valores
        for i in range(8):
            for j in range(8):
                cell_value = value_map[i][j]
                if board.get((i, j)) == player.label:
                    score += cell_value  # Aumentar puntaje para el jugador
                elif board.get((i, j)) == opponent.label:
                    score -= cell_value  # Disminuir puntaje para el oponente

        return score  # Debería ser positivo si el jugador tiene una ventaja


class MobilityHeuristic(StudentHeuristic):
    def get_name(self) -> str:
        return "mobility"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        # A mayor número de movimientos posibles, mayor ventaja estratégica
        player_moves = len(state.get_possible_moves())
        opponent_moves = len(state.get_opponent_moves())

        return player_moves - opponent_moves


class PieceAdvantageHeuristic(StudentHeuristic):
    def get_name(self) -> str:
        return "piece_advantage"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        # Evaluar el número de fichas de cada jugador
        player = state.player1_turn
        board = state.board
        player_score = 0
        opponent_score = 0

        for row in board:
            for cell in row:
                if cell == player:
                    player_score += 1
                elif cell == -player:
                    opponent_score += 1

        return player_score - opponent_score

