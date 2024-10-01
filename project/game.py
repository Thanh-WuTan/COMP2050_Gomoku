"""
This module contains the game logic for Gomoku. 
"""
from typing import List, Optional
from abc import ABC, abstractmethod

class Game(ABC):
    
    @abstractmethod
    def empty_cells(self, state: Optional[List[List[int]]]) -> List[List[int]]:
        """
        Get a list of empty cells for the GIVEN state. If state is None, return the list of empty cells for the CURRENT state.
        :param state: the state of the current board
        :return: a list of empty cells
        """
        pass
    
    @abstractmethod
    def print_board(self):
        """
        Visualization of current board state.
        """
        pass
    
    @abstractmethod
    def init_board(self):
        """
        Draw the initial board and show the game info.
        """
        pass
    
    @abstractmethod
    def valid_move(self, x: int, y: int) -> bool:
        """
        Check if the cell (x,y) is a valid move.
        :param x: X coordinate
        :param y: Y coordinate
        :return: True if the move is valid
        """
        pass
    
    @abstractmethod
    def set_move(self, x: int, y: int, player_letter: str) -> bool:
        """
        Set the move on board, if the coordinates are valid.
        :param x: X coordinate
        :param y: Y coordinate
        :param player_letter: 'X' or 'O
        :return: True if the move is set successfully
        """
        pass

    @abstractmethod
    def wins(self, player_letter: str, state: Optional[List[List[int]]]) -> bool:
        """
        This function tests if a specific player wins in a GIVEN or CURRENT state (if state is None).
        :param state: the state of the current board
        :param player_letter: 'X' or 'O
        :return: True if the player wins
        """
        pass
        
    @abstractmethod
    def game_over(self) -> bool:
        """
        This function test if the game is over for the current state
        :return: True if the game is over (either a player wins or the game is draw)
        """
        pass
    
    @abstractmethod
    def restart(self) -> None:
        """
        This function restarts the game
        """
        pass
    
    @abstractmethod
    def copy(self) -> 'Game':
        """
        This function returns a copy of the current game
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        return "Game"
    
class Gomoku(Game):
    def __init__(self, size=15, board_state = None, last_move = (-1, -1)):  
        self.size = size
        if board_state == None:
            self.board_state = [[None for _ in range(self.size)] for _ in range(self.size)]
        else:
            self.board_state = board_state  
        self.last_move = last_move
        self.win_combo = []
        self.curr_player = 'X'

    def print_board(self):
        height = len(self.board_state)
        width = len(self.board_state[0])

        print("\n  ", end="")
        for x in range(width):
            print("{0:6d}".format(x), end='')
        print('\r\n')
        for i in range(height):
            print("{0:3d}  ".format(i), end='')
            for j in range(width):
                if self.board_state[i][j] == None:
                    print('-'.center(6), end='')
                elif self.board_state[i][j] == 'X':
                    print('X'.center(6), end='')
                elif self.board_state[i][j] == 'O':
                    print('O'.center(6), end='')
                else:
                    raise ValueError("Invalid value in the board")
            print('\n')

    def init_board(self):
        print("\nType 'row,column' to select move.\n", end="  ")

    def empty_cells(self, state=None):
        if state is None:
            state = self.board_state  # Use current state if not provided
            
        cells = []
        for x in range(self.size):
            for y in range(self.size):
                if state[x][y] == None:
                    cells.append([x, y])
        return cells

    def valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board_state[x][y] == None

    def set_move(self, x, y, player_letter):
        if self.valid_move(x, y):
            self.board_state[x][y] = player_letter
            self.last_move = (x, y)
            return True
        return False

    def wins(self, player_letter, state=None):
        # Check if the player wins in the given state, based on the last move
        if state is None:
            state = self.board_state
            
        last_x, last_y = self.last_move
        if self.last_move == None or state[last_x][last_y] != player_letter:
            return False
        
        # four directions: vertical, horizontal, two diagonals
        directions = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]
        
        for axis in directions:
            axis_count = 1
            for (xdirection, ydirection) in axis:
                axis_count += self.direction_count(last_x, last_y, xdirection, ydirection, player_letter, state)
            if axis_count >= 5:
                self.win_combo = self.get_win_combo(last_x, last_y, axis)
                return True
        return False
    
    def direction_count(self, x, y, xdirection, ydirection, letter, state):
        # Count the number of consecutive pieces in a certain direction
        count = 0
        for step in range(1, 5):  # look four more steps on a certain direction
            if xdirection != 0 and (x + xdirection * step < 0 or x + xdirection * step >= self.size):
                break
            if ydirection != 0 and (y + ydirection * step < 0 or y + ydirection * step >= self.size):
                break
            if state[x + xdirection * step][y + ydirection * step] == letter:
                count += 1
            else:
                break
        return count 
    
    def get_win_combo(self, x, y, axis):
        # Get the winning combo based on the last move
        combo = [(x, y)]
        for (xdirection, ydirection) in axis:
            for step in range(1, 5):
                if xdirection != 0 and (x + xdirection * step < 0 or x + xdirection * step >= self.size):
                    break
                if ydirection != 0 and (y + ydirection * step < 0 or y + ydirection * step >= self.size):
                    break
                if self.board_state[x + xdirection * step][y + ydirection * step] == self.board_state[x][y]:
                    combo.append((x + xdirection * step, y + ydirection * step))
                else:
                    break
        return sorted(combo)

    def game_over(self):
        # Check if the game is over (either a player wins or the board is full)
        return self.wins('X') or self.wins('O') or len(self.empty_cells()) == 0
    
    def restart(self):
        self.board_state = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.last_move = (-1, -1)
        self.win_combo = []
        self.curr_player = 'X'
    
    def copy(self):
        new_game = Gomoku(self.size)
        new_game.board_state = [row[:] for row in self.board_state]
        new_game.last_move = self.last_move
        new_game.win_combo = self.win_combo.copy()
        new_game.curr_player = self.curr_player
        return new_game
    
    def get_sequences(self):
        sequences = []
        # row:
        for row in range(self.size):
            for col in range(self.size - 5 + 1):
                seq = []
                for i in range(5):
                    seq.append((self.board_state[row][col + i],  (row, col + i), (0, 1)))
                sequences.append(seq)
        # col:
        for col in range(self.size):
            for row in range(self.size - 5 + 1):
                seq = []
                for i in range(5):
                    seq.append((self.board_state[row + i][col],  (row + i, col), (1, 0)))
                sequences.append(seq)
        # diagonals 1
        for row in range(self.size -  5 + 1):
            for col in range(self.size - 5 + 1):
                seq = []
                for i in range(5):
                    seq.append((self.board_state[row + i][col + i], (row + i, col + i), (1, 1)))
                sequences.append(seq)
        # diagonals 2
        for row in range(self.size - 5 + 1):
            for col in range(4, self.size):
                seq = []
                for i in range(5):
                    seq.append((self.board_state[row + i][col - i],  (row + i, col - i), (1, -1)))
                sequences.append(seq)
        return sequences
    
    def __str__(self) -> str:
        return "Gomoku"
    