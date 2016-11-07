# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.
from operator import itemgetter
from random import randint
# Submission Class 1
class OpenMoveEvalFn():
    """Evaluation function that outputs a
    score equal to how many moves are open
    for the active player."""
    def score(self, game, maximizing_player):
        # TODO: finish this function!
        # print game.print_board()
        # print 'my moves: '
        if game.move_count == 0:
            return game.height*game.width
        if maximizing_player:
            # print game.get_legal_moves().__len__()
            eval_fn =  game.get_legal_moves().__len__()
        else:
            # print game.get_opponent_moves().__len__()
            eval_fn =  game.get_opponent_moves().__len__()

        # print eval_fn
        # print '*'*20

        return eval_fn

# Submission Class 2
class CustomEvalFn():

    """Custom evaluation function that acts
    however you think it should. This is not
    required but highly encouraged if you
    want to build the best AI possible."""
    def score(self, game, maximizing_player):
        # TODO: finish this function!)
        # print game.print_board()
        curr_moves = game.get_legal_moves()

        if maximizing_player:
            opp_moves = len(game.get_opponent_moves())
            # opp_prev_moves = len(self.prevGame.get_legal_moves())
            plyr_moves = len(game.get_legal_moves())

            # plyr_prev_moves = len(self.prevGame.get_opponent_moves())
        else:
            opp_moves = len(game.get_legal_moves())
            plyr_moves = len(game.get_opponent_moves())
            # opp_prev_moves = len(self.prevGame.get_legal_moves())

        if game.move_count <=2:
            return plyr_moves
            #this is the first move, any player can go anywhere
        elif plyr_moves == 0 and not opp_moves == 0:
                #no moves for me, i lose
            return float("-inf")
        elif opp_moves == 0 and not plyr_moves ==0:
            #no moves for opp, i win
            return float("inf")
        elif opp_moves == plyr_moves and plyr_moves == 0:
            #this is a tie
            return -5
        else:
            return plyr_moves - opp_moves*.5
        # elif opp_moves>plyr_moves:
        #     return float(opp_moves)/plyr_moves + (opp_prev_moves - opp_moves)
        # elif plyr_moves<opp_moves:
        #     return -1*float(opp_moves/plyr_moves) + (opp_prev_moves - opp_moves)


# Submission Class 3
class CustomPlayer():

    # TODO: finish this class!
    """Player that chooses a move using
    your evaluation function and
    a depth-limited minimax algorithm
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move
    in less than 1000 milliseconds."""
    def __init__(self, search_depth=9, eval_fn=OpenMoveEvalFn()):
        # if you find yourself with a superior eval function, update the
        # default value of `eval_fn` to `CustomEvalFn()`
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.gameboard = None
        self.num_moves = 0
        self.old_games = {'wins':[],'losses':[]}
        self.move_history = []
        self.pos = None
        self.opp_pos = None
        self.max_depth = 49
        self.middle = [(3,0),(3,1),(3,2),(3,3)]
        self.leftT = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        self.rightT = [(0,4),(0,5),(0,6),(1,4),(1,5),(1,6),(2,4),(2,5),(2,6)]
        self.leftB = [(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),(6,0),(6,1),(6,2)]

    def infiniteTime(self):
        return float("inf")

    # def getQuadrant(self,move):

    def rand_move(self, game, legal):
        numMoves = len(legal)
        move = legal[randint(0,numMoves-1)]
        return move
    def first_move(self,game,legal_moves):
        if game.__player_1__ == self and (3,3) in legal_moves:
            self.pos = (3,3) #middle of the board
            return (3,3)

    def move(self, game, legal_moves, time_left):
        self.time_left = time_left
        moves_left = len(legal_moves)
        game_moves = game.move_count
        # blank = len(game.get_blank_spaces())
        # if self.search_depth > blank:
        #     self.search_depth = blank

        if game.move_count == 0:
            return self.first_move(game,legal_moves)
        if moves_left == 0:
            return (-1,-1)
        elif moves_left == 1:
            return legal_moves[0]
        elif len(game.get_opponent_moves()) == 0:
            return self.rand_move(game,legal_moves)
        # elif moves_left <= 3:
        #     depth = 5
        # else:
        #     depth = self.search_depth

        best_move, utility = self.alphabeta_id(game, self.search_depth)
        #retain current position
        self.pos = best_move

        print best_move , utility

        return best_move


    def pick_direction(self,game,pos):
        r , c = pos


    def surrounded_by(self,pos,blank):
        r,c = pos
        directions = [ (-1, -1), (-1, 0),
                       (-1, 1),(0,-1),(0,1),
                       (1, -1), (1, 0),
                       (1, 1)]
        invalid_moves = [(r+dr,c+dc) for dr, dc in directions
                if not (r+dr, c+dc) in blank]

        return len(invalid_moves)

    def num_col_invalid(self, width, pos, blank):
        r , c = pos
        sum = 0
        for i in range(0,width):
            if not (i,c) in blank:
                sum +=1

        return sum
    def num_row_invalid(self, height, pos, blank):
        r , c = pos
        sum = 0
        for i in range(0,height):
            if not (r,c) in blank:
                sum +=1

        return sum

    def sort_by_utility(self,moves,maximizing_player):
        sorted(moves.items(), key=itemgetter(1),reverse=maximizing_player)
        result = []
        for k in moves.keys():
            result.append(k)
        return result

    def get_my_moves(self, game, maximizing_player):
        if maximizing_player:
            moves = game.get_legal_moves()
        else:
            moves = game.get_opponent_moves()

        return moves


    def branching(self,game, legal_moves, maximizing_player):
        num_moves = 0
        moves = {}
        for move in legal_moves:
            new_game = game.forecast_move(move)
            num_moves += len(self.get_my_moves(new_game,maximizing_player))
            moves[move] = self.eval_fn.score(new_game,maximizing_player)


        return ((num_moves/len(legal_moves)),self.sort_by_utility(moves,maximizing_player))

    def utility(self, game, maximizing_player):

        if game.is_winner(self):
            if maximizing_player:
                return float("inf")
            else:
                return float("-inf")
        elif game.is_opponent_winner(self):
            #game is a loser
            if maximizing_player:
                return float("-inf")
            else:
                return float("inf")
        elif maximizing_player:
            if len(game.get_legal_moves())==0:
                return float("inf")
        elif not maximizing_player:
            if len(game.get_opponent_moves()) == 0:
                return float("-inf")

        return self.eval_fn.score(game, maximizing_player)

    def minmax(self,game,depth,maximizing_player=True):
        best_move = self.rand_move(game,game.get_legal_moves())
        utility = self.utility(game.forecast_move(best_move))
        if game.__player_1__ == self:
            print "Player 1: "
        else:
            print "Player 2:"
        best = []
        #iteraative deepening

        for i in range(1,depth,2):
            if self.time_left() < 100 or utility == float("inf"):
                break;
            max_move, util = self.minimax(game,i, maximizing_player)

            if not util == float("-inf"):
                best_move = max_move
                utility = util
            # print curr_depth , " , " , self.time_left(), ", ", best_move, " : " , utility
        # print 'broke loop'
        return  best_move , utility

    def minimax(self, game, depth, maximizing_player=True):
        # TODO: finish this function!
        # print "minm"
        score = self.utility(game, maximizing_player)

        #if time left, or at depth, return board score
        if depth==0 or self.time_left()< 100:
            return None , score
        #if value is an endgame value, return board score
        if abs(score) == float("inf"):
            return None, score

        moves = game.get_legal_moves()


        best_move = None
        best_val = float("-inf") if maximizing_player else float("inf")
        next_depth = depth-1

        for next_move in moves:
            new_board = game.forecast_move(next_move)
            curr_move , curr_val = self.minimax(new_board, next_depth, not maximizing_player)
            # print next_move , curr_val
            if (not maximizing_player and curr_val <= best_val) or (maximizing_player and curr_val>=best_val):
                best_move = next_move
                best_val = curr_val
                if best_val == float("inf"):
                    return best_move , best_val
        return best_move, best_val

    def reflect_player(self, move):

        r, c = move
        print r,c
        mr = 3
        mc = 3
        print mr , "-" , r
        print mc , "-" , c

        nr = mr - r
        nc = mc - c
        print nr , nc
        cr, cc = self.pos
        print (cc + nc,cr + nr)
        return (cc + nc,cr + nr)


        # (5,2) == (4,1)
    def ab(self, game, depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        score = self.utility(game, maximizing_player)
        if depth == 0 or self.time_left() < 25:
            return None, score

        moves = game.get_legal_moves()

        best_move = None
        if len(moves) > 0:
            best_move = self.rand_move(game,moves)

        if maximizing_player:
            best_val = float("-inf")
            for next_move in moves:
                new_game = game.forecast_move(next_move)
                curr_move, curr_val = self.ab(new_game, depth - 1, alpha, beta, False)

                if alpha >= beta:
                    break

                if curr_val > best_val:
                    best_val = curr_val
                    alpha = curr_val
                    best_move = next_move
        else:
            best_val = float("inf")
            for next_move in moves:
                new_game = game.forecast_move(next_move)
                curr_move, curr_val = self.ab(new_game, depth - 1, alpha, beta, True)

                if beta <= alpha:
                    break

                if curr_val < best_val:
                    best_val = curr_val
                    beta = curr_val
                    best_move = next_move

        return best_move, best_val
    def alphabeta_id(self,game,depth,alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        best_move = self.rand_move(game,game.get_legal_moves())
        utility = self.utility(game.forecast_move(best_move),maximizing_player)
        if game.__player_1__ == self:
            print "Player 1: "
        else:
            print "Player 2:"
        best = []
        #iteraative deepening
        for i in range(1,depth):
            # print "time left", self.time_left()
            #hold best val for current depth
            if self.time_left() < 100 or utility == float("inf"): #not enough time left for another iteration
                return  best_move , utility
            max_move, util = self.ab(game,i, alpha, beta, maximizing_player)

            if not util == float("-inf"):
                best_move = max_move
                utility = util

        return best_move , utility
