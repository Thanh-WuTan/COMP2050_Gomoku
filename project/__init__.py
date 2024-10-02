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
        elif player == 'alphabeta':
            agent = GMK_AlphaBetaPlayer(letter)
        elif player == 'mcts_v1':
            agent = GMK_NaiveMCTS(letter)
        elif player == 'mcts_v2':
            agent = GMK_BetterMCTS(letter)
        elif player == 'qplayer_v1':
            agent = GMK_TabularQPlayer(letter, size=args.size)
        elif player == 'qplayer_v2':
            agent = GMK_ApproximateQPlayer(letter, size=args.size)
        elif player == 'beginner':
            agent = GMK_Beginner(letter)
        elif player == 'intermediate':
            agent = GMK_Intermediate(letter)
        elif player == 'advanced':
            agent = GMK_Advanced(letter)
        elif player == 'master':
            agent = GMK_Master(letter)
        elif player == 'reflex':
            agent = GMK_Reflex(letter)
        else:
            raise ValueError(f"{player.capitalize()} player is not defined for {args.game.capitalize()}.")
    else:
        raise ValueError("Invalid game") 
    return agent
 