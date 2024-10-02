import numpy as np
import math

from .reflex import *
from .alphabeta import *
from ..player import Player
from ..game import Gomoku
from copy import deepcopy

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 15

import random
SEED = 2024
random.seed(SEED)

class TreeNode():
    def __init__(self, game_state: Gomoku, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0
   
    def select(self):
        if self.is_leaf_node():
            return self
        return self.best_child().select()
   
    def expand(self) -> 'TreeNode':
        ab_agaent = GMK_AlphaBetaPlayer(self.player)
        promissing_moves = ab_agaent.promising_next_moves(self.game_state, self.player)
        for move in promissing_moves:
            child_game_state = self.game_state.copy()
            child_game_state.set_move(move[0], move[1], self.player)
            child_player = 'X' if self.player == 'O' else 'O'
            child_node = TreeNode(child_game_state, child_player, parent=self, parent_action=move)
            self.children.append(child_node)
   
    def simulate(self) -> int:
        player_letter = self.player
        opponent_letter = 'X' if player_letter == 'O' else 'O'
       
        curr_letter = player_letter
        simulate_game = self.game_state.copy()
        limit_depth = 10
        while True:            
            if simulate_game.wins(player_letter):
                return WIN
            elif simulate_game.wins(opponent_letter):
                return LOSE
            elif len(simulate_game.empty_cells()) == 0:
                return DRAW
            elif limit_depth == 0:
                ab_agent = GMK_AlphaBetaPlayer(curr_letter)
                evaluate = ab_agent.evaluate(deepcopy(simulate_game), curr_letter)
                if evaluate > 0:
                    if curr_letter == player_letter:
                        return WIN
                    else:
                        return LOSE
                elif evaluate == 0:
                    return DRAW
                else:
                    if curr_letter == player_letter:
                        return LOSE
                    else:
                        return WIN
            else:  
                reflex_agent = GMK_Reflex(curr_letter)
                move = reflex_agent.get_move(simulate_game)
                simulate_game.set_move(move[0], move[1], curr_letter)
                curr_letter = 'X' if curr_letter == 'O' else 'O'
                limit_depth-= 1
   
    def backpropagate(self, result):
        if self.parent:
            self.parent.backpropagate(-result)
        self.N+= 1
        self.Q+= result
           
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
   
    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()
   
    def best_child(self) -> 'TreeNode':
        return max(self.children, key=lambda c: c.ucb())
   
    def ucb(self, c=math.sqrt(2)) -> float:
        if self.N == 0:
            return float('inf')
        return self.Q / self.N + c * np.sqrt(np.log(self.parent.N) / self.N)
    
class GMK_BetterMCTS(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations
    
    def get_move(self, game: Gomoku):
        if game.last_move == (-1, -1):
            return (game.size // 2, game.size // 2)
        cnt = 0
        for row in range(game.size):
            for col in range(game.size):
                cnt+= game.board_state[row][col] != None
        if cnt == 1:
            return (game.last_move[0], game.last_move[1] - 1)
        mtcs = TreeNode(game, self.letter)
        for num in range(self.num_simulations):
            leaf = mtcs.select()
            if not leaf.is_terminal_node():
                leaf.expand()
            result = leaf.simulate()
            leaf.backpropagate(-result)
            
        best_child = max(mtcs.children, key=lambda c: c.N)
        return best_child.parent_action
    
    def __str__(self) -> str:
        return "Better MCTS Player"
    