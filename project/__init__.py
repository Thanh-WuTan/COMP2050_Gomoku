from .game import * 
from .gomoku import *
from .player import *
 
def Game(args):
    if args.game == 'gomoku':
        game = Gomoku(args.size)
    else:
        raise ValueError("Invalid game")
    return game
 
def Player(args, player, letter):
    if args.game == 'gomoku':
        if player == 'random':
            agent = RandomPlayer(letter)
        elif player == 'human':
            agent = GMK_HumanPlayer(letter)
        elif player == 'beginner':
            agent = GMK_Beginner(letter)
        elif player == 'intermediate':
            agent = GMK_Intermediate(letter)
        elif player == 'advanced':
            agent = GMK_Advanced(letter)
        elif player == 'master':
            agent = GMK_Master(letter)
        else:
            raise ValueError(f"{player.capitalize()} player is not defined for {args.game.capitalize()}.")
    else:
        raise ValueError("Invalid game") 
    return agent
 