from ..player import Player
from ..game import Gomoku
from typing import List, Tuple, Union

from .reflex import *


import math
import random
SEED = 2024
random.seed(SEED)

DEPTH = 2 # Define the depth of the search tree.

class GMK_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.depth = DEPTH 
       
    
    def get_move(self, game: Gomoku):
        cnt = 0
        for row in range(game.size):
            for col in range(game.size):
                if game.board_state[row][col] != None:
                    cnt+= 1
            
        if cnt == 0:
            move = (game.size // 2, game.size // 2)
            return move
        # Alpha-Beta Pruning: Initialize alpha to negative infinity and beta to positive infinity
        alpha = -math.inf
        beta = math.inf
        
        choice = self.minimax(game, self.depth, self.letter, alpha, beta)
        move = [choice[0], choice[1]]
        return move

    def minimax(self, game, depth, player_letter, alpha, beta) -> Union[List[int], Tuple[int]]:
        """
        AI function that chooses the best move with alpha-beta pruning.
        :param game: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9)
        :param player_letter: value representing the player
        :param alpha: best value that the maximizer can guarantee
        :param beta: best value that the minimizer can guarantee
        :return: a list or a tuple with [best row, best col, best score]
        """
        if player_letter == self.letter:
            best = (-1, -1, -100000000000)  # Max Player
        else:
            best = (-1, -1, +100000000000)  # Min Player

        if depth == 0 or game.game_over():
            score = self.evaluate(game, player_letter)
            return (-1, -1, score)
        
      
        promising_moves = self.promising_next_moves(game, player_letter)
        
        for cell in promising_moves:
            
            x, y = cell[0], cell[1]
            game.board_state[x][y] = player_letter
            other_letter = 'X' if player_letter == 'O' else 'O'
            score = self.minimax(game, depth - 1, other_letter, alpha, beta)
            game.board_state[x][y] = None
            score = (x, y, score[2])
            if player_letter == self.letter:  # Max player
                if score[2] > best[2]:
                    best = score
                alpha = max(alpha, best[2])
                if beta <= alpha:
                    
                    break
            else:  # Min player
                if score[2] < best[2]:
                    best = score
                beta = min(beta, best[2])
                if beta <= alpha:
                    break 
    
        return best
    
    def evaluate(self, game, player_letter) -> float:
        """
        Define a heuristic evaluation function for the given state when leaf node is reached.
        :return: a float value representing the score of the state
        """
        oponent_letter = 'X' if player_letter == 'O' else 'O'
        if game.wins(player_letter):
            return 10000000000
        if game.wins(oponent_letter):
            return -10000000000
        
        reflex_agent = GMK_Reflex(player_letter)
        
        sequences = game.get_sequences()
        if reflex_agent.policy_one(sequences) != None:
            return 10000000000
        
        if len(reflex_agent.policy_two(sequences)) >= 2:
            return -10000000000
        
        if len(reflex_agent.policy_three(sequences, game)) > 0:
            return 10000000000

        if len(reflex_agent.policy_four(sequences, game)) > 0:
            return 10000000000
        
        if len(reflex_agent.policy_five(sequences, game)) > 0:
            return 10000000000

        tmp = 0
        tmp+= len(reflex_agent.policy_seven(sequences, game)) 
        tmp+= len(reflex_agent.policy_eight(sequences, game))
        tmp+= len(reflex_agent.policy_nine(sequences, game))
        if tmp > 1:
            return -10000000000
        
        if len(reflex_agent.policy_ten(sequences, game)) > 0:
            return 10000000000
        

        if len(reflex_agent.policy_eleven(sequences, game)) > 1:
            return -10000000000

        score = 0
        for seq in sequences:
            potential = True
            cnt = 0
            for cell in seq:
                if cell[0] == None:
                    continue
                if cell[0] != player_letter:
                    potential = False
                    break
                cnt+= 1
            if potential:
                score+= 10 ** cnt

            potential = True
            cnt = 0
            for cell in seq:
                if cell[0] == None:
                    continue
                if cell[0] != oponent_letter:
                    potential = False
                    break
                cnt+= 1
            if potential:
                score-= 10 ** (cnt + 1)

        return score
    
    def promising_next_moves(self, game, player_letter) -> List[Tuple[int]]:
        """
        Find the promosing next moves to explore, so that the search space can be reduced.
        :return: a list of tuples with the best moves
        """
        pro_moves = []
        ######### YOUR CODE HERE #########
        reflex_agent = GMK_Reflex(player_letter)
        
        
        sequences = game.get_sequences()
        
        move = reflex_agent.policy_one(sequences) 
        if move != None:
            pro_moves.append(move)
            return pro_moves
        
        moves = reflex_agent.policy_two(sequences)
        if len(moves) > 0:
            return moves
        
        moves = reflex_agent.policy_three(sequences, game)
        for move in moves:
            pro_moves.append(move)
        
        moves = reflex_agent.policy_four(sequences, game)
        for move in moves:
            pro_moves.append(move)

        moves = reflex_agent.policy_five(sequences, game)
        for move in moves:
            pro_moves.append(move)
        
        if len(pro_moves) > 0:
            return pro_moves
        
    

        moves = reflex_agent.policy_seven(sequences, game)
        for move in moves:
            pro_moves.append(move)
        
        moves = reflex_agent.policy_eight(sequences, game)
        for move in moves:
            pro_moves.append(move)
        
        moves = reflex_agent.policy_nine(sequences, game)
        for move in moves:
            pro_moves.append(move)

        if len(pro_moves) > 0:
            return pro_moves

        moves = reflex_agent.policy_ten(sequences, game)
        if len(moves) > 0:
            return moves

        moves = reflex_agent.policy_eleven(sequences, game)
        if len(moves) > 0:
            return moves
         
        

        move = reflex_agent.policy_default(sequences, game)
        pro_moves.append(move)

        for cell in game.empty_cells():
            cnt = reflex_agent.heuristic(game.board_state, player_letter, cell[0], cell[1]) - 4
            cnt = max(cnt, reflex_agent.heuristic(game.board_state, 'X' if player_letter == 'O' else 'O', cell[0], cell[1])) - 4
            if cnt > 0:
                pro_moves.append((cell[0], cell[1]))
      
        return pro_moves

        # best_cnt = 0
        # tmp_arr = []
        # for cell in game.empty_cells():
        #     cnt = reflex_agent.heuristic(game.board_state, player_letter, cell[0], cell[1]) - 4
        #     cnt = max(cnt, reflex_agent.heuristic(game.board_state, 'X' if player_letter == 'O' else 'O', cell[0], cell[1])) - 4
        #     best_cnt = max(best_cnt, cnt)
        #     if cnt > 0:
        #         tmp_arr.append((cell[0], cell[1], cnt))
        # for cell in tmp_arr:
        #     if abs(best_cnt - cell[2]) <= 16:
        #         pro_moves.append((cell[0], cell[1]))
        # return pro_moves
    
    def get_hash_board(self, game):
        hash = ""
        for row in range(game.size):
            for col in range(game.size):
                if game.board_state[row][col] == None:
                    hash+= '0'
                elif game.board_state[row][col] == 'X':
                    hash+= '1'
                else:
                    hash+= '2'
        return hash
    
    def __str__(self):
        return "AlphaBeta Player"
