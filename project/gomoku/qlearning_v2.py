import math
from typing import List, Tuple, Union, DefaultDict
from tqdm import tqdm
from ..player import Player
from .reflex import *
from collections import defaultdict
from .bots.intermediate import GMK_Intermediate

import numpy as np
import random
import os
import pickle

NUM_EPISODES = 100
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
SEED = 2024
random.seed(SEED)

class GMK_ApproximateQPlayer(Player):
    def __init__(self, letter, size=15, transfer_player=GMK_Intermediate):
        super().__init__(letter)
        self.opponent = transfer_player
        self.num_episodes = NUM_EPISODES
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EXPLORATION_RATE
        self.weights = defaultdict(float) # Initialize weights to 0
        self.action_history = []
        self.board_size = size
        self.feature_extractor = SimpleExtractor() 
        self.weights['straigth-win'] = 1000
        
        self.weights['lose-in-one-move'] = -100
        
        self.weights['#-of-unblocked-four-player-created-by-the-move'] = 90
        
        self.weights['#-of-unblocked-three-player-created-by-the-move'] = 90

        self.weights['block-aggressive-move'] = 100
        

        self.weights['#-of-unblocked-four-oponent'] = -100
        self.weights['#-of-unblocked-three-oponent'] = -50

        self.weights['heuristic-attack'] = 20
        self.weights['heuristic-defense'] = 10

    def train(self, game, save_filename=None):
        # Main Q-learning algorithm
        opponent_letter = 'X' if self.letter == 'O' else 'O'
        if self.opponent is None:
            opponent = GMK_ApproximateQPlayer(opponent_letter)
        else:
            opponent = self.opponent(opponent_letter)
            
        print(f"Training {self.letter} player for {self.num_episodes} episodes...")
        game_state = game.copy()
        
        for _ in tqdm(range(self.num_episodes)):               
            game_state.restart()
            opponent.action_history = []
            
            current_player = self if self.letter == 'X' else opponent 
            next_player = self if self.letter == 'O' else opponent
            while True:                
                if isinstance(current_player, GMK_ApproximateQPlayer):     
                    action = current_player.choose_action(game_state)
                    state = copy.deepcopy(game_state.board_state)
                    current_player.action_history.append((state, action)) 
                else:
                    action = current_player.get_move(game_state)
                
                next_game_state = game_state.copy()
                next_game_state.set_move(action[0], action[1], current_player.letter)
                
                if next_game_state.game_over():
                    reward = 1 if next_game_state.wins(current_player.letter) else -1 if next_game_state.wins(next_player.letter) else 0
                    if isinstance(current_player, GMK_ApproximateQPlayer):
                        current_player.update_rewards(reward)
                    if isinstance(next_player, GMK_ApproximateQPlayer):
                        next_player.update_rewards(-reward)
                    break
                else: 
                    current_player, next_player = next_player, current_player
                    game_state = next_game_state    

            self.letter = 'X' if self.letter == 'O' else 'O'
            opponent.letter = 'X' if opponent.letter == 'O' else 'O'  
            self.action_history = []
        
        print("Training complete. Saving training weights...")
        if save_filename is None:
            save_filename = f'{self.board_size}x{self.board_size}_{NUM_EPISODES}.pkl'
        self.save_weight(save_filename)
    
    def update_rewards(self, reward: float):
        """
        Given the reward at the end of the game, update the weights for each state-action pair in the game with the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * Q(s', a') - Q(s, a)) * f_i(s, a)

        * We need to update the Q-values for each state-action pair in the action history because the reward is only received at the end.
        * Make a call to update_q_values() for each state-action pair in the action history.
        """
        for t in range(len(self.action_history) - 1, -1, -1):
            state, action = self.action_history[t]
            if t == len(self.action_history) - 1:
                next_state = None
            else:
                next_state, _ = self.action_history[t + 1]
            self.update_q_values(state, action, next_state, reward)
            reward *= self.gamma

    def choose_action(self, game) -> Union[List[int], Tuple[int, int]]:
        """
        Choose action with ε-greedy strategy.
        If random number < ε, choose random action.
        Else choose action with the highest Q-value.
        :return: action
        """
        state = copy.deepcopy(game.board_state)
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(game.empty_cells())
        else:
            # Choose the action with the highest Q-value
            max_q_value = -math.inf
            best_action = None
            for action in game.empty_cells():
                q_value = self.q_value(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            return best_action


    def update_q_values(self, state, action, next_state, reward):
        """
        Given (s, a, s', r), update the weights for the state-action pair (s, a) using the TD update rule:
            for weight w_i of feature f_i for (s, a):
                w_i = w_i + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)) * f_i(s, a)
        :return: None
        """
        features = self.feature_vector(state, action)
        q_state_action = self.q_value(state, action)
        if next_state is not None:
            max_q_next = self.max_q_value(next_state)
        else:
            max_q_next = 0
        for feature_name, feature_value in features.items():
            self.weights[feature_name] += self.learning_rate * (reward + self.gamma * max_q_next - q_state_action) * feature_value

    def feature_vector(self, state, action) -> DefaultDict[str, float]:
        """
        Extract the feature vector for a given state-action pair.
        :return: feature vector
        """
        return self.feature_extractor.get_features(copy.deepcopy(state), action, self.letter)

    def max_q_value(self, state):
        max_q_value = -math.inf
        for action in self.empty_cells(state):
            q_value = self.q_value(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
        return max_q_value

    def q_value(self, state, action) -> float:
        """
        Compute the Q-value for a given state-action pair as the dot product of the feature vector and the weight vector.
        :return: Q-value
        """
        q_value = 0
        features = self.feature_vector(state, action)
        for feature_name, feature_value in features.items():
            q_value += self.weights[feature_name] * feature_value
        return q_value
    
    def save_weight(self, filename):
        """
        Save the weights of the feature vector.
        """
        path = 'project/gomoku/q_weights'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/{filename}', 'wb') as f:
            pickle.dump(dict(self.weights), f)

    def load_weight(self, filename):
        """
        Load the Q-table.
        """
        path = 'project/gomoku/q_weights'
        if not os.path.exists(f'{path}/{filename}'):
            raise FileNotFoundError(f"Weight file '{filename}' not found.")
        with open(f'{path}/{filename}', 'rb') as f:
            dict_weights = pickle.load(f)
            self.weights.update(dict_weights)

    def get_move(self, game):
        if game.last_move == (-1, -1):
            return (game.size // 2, game.size // 2)
        self.epsilon = 0  # No exploration
        # print(self.letter)
        # for row in game.board_state:
        #     print(row) 
        # print("-------")
        # for move in game.empty_cells():
        #     print("----")
        #     print(move, self.q_value(game.board_state, move))
        #     features = self.feature_vector(game.board_state, move)
        #     for feature_name, feature_value in features.items():
        #         print(feature_name, self.weights[feature_name], feature_value)
        return self.choose_action(game)
    
    def empty_cells(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        """
        Return a list of empty cells in the board.
        """
        return [(x, y) for x in range(len(board)) for y in range(len(board[0])) if board[x][y] is None]

    def __str__(self):
        return "Approximate Q-Learning Player"

########################### Feature Extractor ###########################
from abc import ABC, abstractmethod
import copy

class FeatureExtractor(ABC):
    @abstractmethod
    def get_features(self, state: List[List[str]], move: Union[List[int], Tuple[int]], player: str) -> DefaultDict[str, float]:
        """
        :param state: current board state
        :param move: move taken by the player
        :param player: current player
        :return: a dictionary {feature_name: feature_value}
        """
        pass

class IdentityExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        """
        Return 1.0 for all state action pair.
        """
        feats = defaultdict(float)
        key = self.hash_board(state)
        feats[(key, tuple(move))] = 1.0
        return feats
    
    def hash_board(self, board):
        key = ''
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'X':
                    key += '1'
                elif board[i][j] == 'O':
                    key += '2'
                else:
                    key += '0'
        return key


class SimpleExtractor(FeatureExtractor):
    def get_features(self, state, move, player):
        """
        features: #-of-unblocked-three-player, #-of-unblocked-three-opponent,
                  #-of-unblocked-four-player, #-of-unblocked-four-opponent,
                  #-of-blocked-four-player, #-of-blocked-four-opponent,
                  #-of-cross-pattern-player, #-of-cross-pattern-opponent
        """
        opponent = 'X' if player == 'O' else 'O'
        size = len(state)
        
        game = Gomoku(size = size, board_state = deepcopy(state))
        
        game_tmp = Gomoku(size = size, board_state = deepcopy(state))

        game.set_move(move[0], move[1], player)
        
        sequences = game.get_sequences()
        
        x, y = move
        state = np.array(state)
        state[x][y] = player

        feats = defaultdict(float)


        feats['straigth-win'] = 1 if game.wins(player) == True else 0
        
        if feats['straigth-win'] == 0:
            feats['lose-in-one-move'] = self.lose_in_one_move(sequences, opponent)
        
        if feats['lose-in-one-move'] == 0:
            feats['#-of-unblocked-four-player-created-by-the-move'] = self.count_open_four_created_by_the_move(sequences, player, state, size, move)
            feats['#-of-unblocked-three-player-created-by-the-move'] = self.count_open_three_created_by_the_move(sequences, player, state, size, move)            
            feats['block-aggressive-move'] = self.block_aggressive_move(game_tmp, opponent, move[0], move[1])

        feats['#-of-unblocked-four-oponent'] = self.count_open_four(sequences, opponent, state, size)
        feats['#-of-unblocked-three-oponent'] = self.count_open_three(sequences, opponent, state, size)

        feats['heuristic-attack'] = self.heuristic(state, player, move[0], move[1])
        feats['heuristic-defense'] = self.heuristic(state, opponent, move[0], move[1])
        

        return feats
    
    def heuristic(self, board, player, row, col):
        modified_board = deepcopy(board)
        modified_board[row][col] = player
        board_size = len(board)

        max_count = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 0
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < board_size and 0 <= c < board_size and modified_board[r][c] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < board_size and 0 <= c < board_size and modified_board[r][c] == player:
                    count += 1
                else:
                    break
            max_count += count
        return max_count

    def count_open_four(self, sequences, opponent, state, size):
        total = 0
        for seq in sequences:
            potential = True
            cnt = 0
            for cell in seq:
                if cell[0] == None:
                    continue    
                if cell[0] != opponent:
                    potential = False
                    break
                cnt+= 1
                 
            if potential == False:
                continue
            if cnt != 4:
                continue
            if cnt == 4:
                total+= 1
            
            cnt = 0 
            for i in range(4):
                cell = seq[i]
                if cell[0] == None:
                    continue 
                cnt+= 1
            if cnt == 4:
                cell = seq[0][1]
                type = seq[0][2]
                if 0 <= cell[0] - type[0] < size and 0 <= cell[1] - type[1]  < size:
                    if state[cell[0] - type[0]][cell[1] - type[1]] == None or state[cell[0] - type[0]][cell[1] - type[1]] == opponent:
                        total+= 1

            cnt = 0
            for i in range(1, 5):
                cell = seq[i]
                if cell[0] == None: 
                    continue 
                cnt+= 1
            if cnt == 4:
                cell = seq[-1][1]
                type = seq[-1][2]
                if 0 <= cell[0] + type[0] < size and 0 <=  cell[1] + type[1] < size:
                    if state[cell[0] + type[0]][cell[1] + type[1]] == None or state[cell[0] + type[0]][cell[1] + type[1]] == opponent:
                        total+= 1
        return total
    
    def count_open_three(self, sequences, opponent, state, size):
        total = 0
        for seq in sequences:
            potential = True
            cnt = 0 
            for cell in seq:
                if cell[0] == None:
                    continue    
                if cell[0] != opponent:
                    potential = False
                    break
                cnt+= 1
                
            if potential == False:
                continue
            if cnt != 3:
                continue
            
            potential = True
            cnt = 0
            for i in range(4):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[0]
            type = cell[2]
            if cell[1][0] - type[0] < 0 or cell[1][1] - type[1] < 0 or cell[1][0] - type[0] >= size or cell[1][1] - type[1] >= size:
                potential = False
            elif state[cell[1][0] - type[0]][cell[1][1] - type[1]] != None and state[cell[1][0] - type[0]][cell[1][1] - type[1]] != opponent:
                potential = False
            if cnt == 3 and potential == True:
                total+= 1
            
            potential = True
            cnt = 0
            for i in range(1, 5):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[4]
            type = cell[2]
            if cell[1][0] + type[0] < 0 or cell[1][1] + type[1] < 0 or cell[1][0] + type[0] >=  size or cell[1][1] + type[1] >= size:
                potential = False
            elif state[cell[1][0] + type[0]][cell[1][1] + type[1]] != None and state[cell[1][0] + type[0]][cell[1][1] + type[1]] != opponent:
                potential = False
            if cnt == 3 and potential == True:    
                total+= 1

        return total

    def block_aggressive_move(self, game, opponent, row, col):
        total = 0
        sequences = game.get_sequences()
        reflex = GMK_Reflex(opponent)
        moves = reflex.policy_three(sequences, game)
        for move in moves:
            if move[0] == row and move[1] == col:
                total+= 1
        
        moves = reflex.policy_four(sequences, game)
        for move in moves:
            if move[0] == row and move[1] == col:
                total+= 1

        moves = reflex.policy_five(sequences, game)
        for move in moves:
            if move[0] == row and move[1] == col:
                total+= 1
            
        moves = reflex.policy_ten(sequences, game)
        for move in moves:
            if move[0] == row and move[1] == col:
                total+= 1
        return total

    def count_open_three_created_by_the_move(self, sequences, player, state, size, move):
        total = 0
        for seq in sequences:
            potential = True
            cnt = 0
            has_the_move = False
            for cell in seq:
                if cell[0] == None:
                    continue    
                if cell[0] != player:
                    potential = False
                    break
                cnt+= 1
                if cell[1][0] == move[0] and cell[1][1] == move[1]:
                    has_the_move = True
            
            if potential == False or has_the_move == False:
                continue
            if cnt != 3:
                continue
             
            potential = True
            cnt = 0
            for i in range(4):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[0]
            type = cell[2]
            if cell[1][0] - type[0] < 0 or cell[1][1] - type[1] < 0 or cell[1][0] - type[0] >= size or cell[1][1] - type[1] >= size:
                potential = False
            elif state[cell[1][0] - type[0]][cell[1][1] - type[1]] != None and state[cell[1][0] - type[0]][cell[1][1] - type[1]] != player:
                potential = False
            if cnt == 3 and potential == True:
                total+= 1 
             
            potential = True
            cnt = 0
            for i in range(1, 5):
                if seq[i][0] != None:
                    cnt+= 1
            cell = seq[4]
            type = cell[2]
            if cell[1][0] + type[0] < 0 or cell[1][1] + type[1] < 0 or cell[1][0] + type[0] >=  size or cell[1][1] + type[1] >= size:
                potential = False
            elif state[cell[1][0] + type[0]][cell[1][1] + type[1]] != None and state[cell[1][0] + type[0]][cell[1][1] + type[1]] != player:
                potential = False
            if cnt == 3 and potential == True:    
                total+= 1 
            
        return total

    def count_open_four_created_by_the_move(self, sequences, player, state, size, move):
        total = 0
        for seq in sequences:
            potential = True
            has_the_move = False
            cnt = 0
            for cell in seq:
                if cell[0] == None:
                    continue    
                if cell[0] != player:
                    potential = False
                    break
                cnt+= 1
                if cell[1][0] == move[0] and cell[1][1] == move[1]:
                    has_the_move = True
            if potential == False or has_the_move == False:
                continue

            if cnt != 4:
                continue

            if cnt == 4:
                total+= 1

            cnt = 0 
            for i in range(4):
                cell = seq[i]
                if cell[0] == None:
                    continue 
                cnt+= 1
            if cnt == 4:
                cell = seq[0][1]
                type = seq[0][2]
                if 0 <= cell[0] - type[0] < size and 0 <= cell[1] - type[1]  < size:
                    if state[cell[0] - type[0]][cell[1] - type[1]] == None or state[cell[0] - type[0]][cell[1] - type[1]] == player:
                        total+= 1

            cnt = 0
            for i in range(1, 5):
                cell = seq[i]
                if cell[0] == None: 
                    continue 
                cnt+= 1
            if cnt == 4:
                cell = seq[-1][1]
                type = seq[-1][2]
                if 0 <= cell[0] + type[0] < size and 0 <=  cell[1] + type[1] < size:
                    if state[cell[0] + type[0]][cell[1] + type[1]] == None or state[cell[0] + type[0]][cell[1] + type[1]] == player:
                        total+= 1
            
        return total

    def lose_in_one_move(self, sequences, oponent):
        for seq in sequences:
            cnt = 0
            potential = True
            for cell in seq:
                if cell[0] == None:
                    continue
                if cell[0] != oponent:
                    potential = False
                    break
                cnt+= 1
            if potential == False:
                continue
            if cnt == 4:
                return 1
        return 0
