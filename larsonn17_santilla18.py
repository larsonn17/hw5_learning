import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #CONSTANTS
    alpha = .2
    e = 2.71828
    neuralSize = 15 #should be updated

    global weightList
    weightList = []

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Dumb Bunny")
        self.depthLimit = 2
        self.bestOverallScore = 0
        self.stateList = []
        self.neuralScoreList = []
        self.deltaList = []
        if weightList == None:
            while size(weightList) < self.NeuralSize: #17, not including workerDist yet
                weightList.append(random.range(0.1, .9))

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    ############
    ##
    #examineGameState
    #
    #Description: called whenever we need to evaluate hypothetical moves to
    #determine which move generates the best possible outcome
    #
    #Parameters:
    #   currentState - the state of the game to be evaluated.
    #
    #Return: a value between 0 1nd 1 describing how good the state is,
    #   higher values indicate moves that are closer to winning the game
    ##
    def examineGameState(self, currentState):

        myInventory = getCurrPlayerInventory(currentState)
        utility = 100.0

        # Store the location of the tunnel, anthill, and food
        tunnel = myInventory.getTunnels()
        my_tunnel_cords = tunnel[0].coords
        my_anthill_coords = myInventory.getAnthill().coords
        my_food_coords = []
        foods = getConstrList(currentState, None, (FOOD,))
        for food in foods:
            if food.coords[1] < 5:
                my_food_coords.append(food.coords)

    # This is where we will incentivize our ants.
        #large negative value for losing queen or worker
        if len(myInventory.ants) < 2:
            utility -= 300

        for ant in myInventory.ants:
            #do not want the queen to be on food or food drop off points
            if ant.type == QUEEN:
                if (ant.coords == my_tunnel_cords or ant.coords == my_anthill_coords or
                    ant.coords == my_food_coords[0] or ant.coords == my_food_coords[1]):
                    utility -= 100
            elif ant.type == WORKER:
                # we need to know which food the ant is closer too
                food_1_distance = approxDist(ant.coords, my_food_coords[0])
                food_2_distance = approxDist(ant.coords, my_food_coords[1])

                ant_to_anthill = approxDist(ant.coords, my_anthill_coords)
                ant_to_tunnel = approxDist(ant.coords, my_tunnel_cords)
                #if ant doesnt have food, encourage it to get food
                if ant.carrying == False:
                    if (food_1_distance < food_2_distance):
                        utility += 100 - (10 * food_1_distance)
                    elif (food_2_distance < food_1_distance):
                        utility += 100 - (10 * food_2_distance)
                        if food_1_distance < 1 or food_2_distance < 1:
                            utility += 100
                #if it does have food, have it find the closest drop off point
                if ant.carrying == True:
                    if (ant_to_tunnel < ant_to_anthill):
                        utility += 200 - (10 * ant_to_tunnel)
                    else:
                        utility += 200 - (10 * ant_to_anthill)
            #score increased by amount of food
            utility += myInventory.foodCount*200
            #scale down to a value between 0 and 1
            utility = utility/2700
        return utility

    #####
    def neuralEvaluation(self, stateList):

        shuffle(stateList)
        # ^ needs to be put in before call to neural evalation so that the same order is present in both this function and the backPropogation function

        for state in stateList:
            #defaults for various scores
            statescore = 0
            numAnts = 1
            queenOffAntHill = 1
            workerCarrying = 1
            #foodDist = [0, 0, 0, 0, 0, 0]   #workerDist from food or tunnel (1-6 tiles)
            #tunnelDist = [0,0,0,0]
            foodCount = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #12 possibilities

            #pull out desired info from each state
            inventory = getCurrPlayerInventory(state)
            for ant in inventory.ants:
                if ant.type == QUEEN and (ant.coords == my_tunnel_cords
                or ant.coords == my_anthill_coords or ant.coords == my_food_coords[0]
                or ant.coords == my_food_coords[1]):
                    queenOffAntHill = 0

                if ant.type == WORKER:
                    if ant.carrying == False:
                        workerCarrying = 0
                        #Test for food distance:
                        #food_1_distance = approxDist(ant.coords, my_food_coords[0])
                        #food_2_distance = approxDist(ant.coords, my_food_coords[1])
                        #if (food_1_distance < food_2_distance):
                        #    dist = food_1_distance
                        #else: dist = food_2_distance
                        #if dist >6: dist = 6
                        #workerDist[dist] = 1
                    #else:
                    #    ant_to_anthill = approxDist(ant.coords, my_anthill_coords)
                    #    ant_to_tunnel = approxDist(ant.coords, my_tunnel_cords)
                    #    if (ant_to_tunnel < ant_to_anthill):
                    #        dist = ant_to_tunnel
                    #    else: dist = ant_to_anthill
                    #    if dist > 6

            if size(inventory.ants) < 2:
                numAnts = 0
            foodCount[inventory.foodCount] = 1


            #sum the weights with the boolean value of each state
            statescore += weight[0]*numAnts
            statescore += weight[1]*queenOffAntHill
            statescore += weight[2]*workerCarrying
            statescore += weight[3]*foodCount[0]
            statescore += weight[4]*foodCount[1]
            statescore += weight[5]*foodCount[2]
            statescore += weight[6]*foodCount[3]
            statescore += weight[7]*foodCount[4]
            statescore += weight[8]*foodCount[5]
            statescore += weight[9]*foodCount[6]
            statescore += weight[10]*foodCount[7]
            statescore += weight[11]*foodCount[8]
            statescore += weight[12]*foodCount[9]
            statescore += weight[13]*foodCount[10]
            statescore += weight[14]*foodCount[11]

            #append to the neural score list
            self.neuralScoreList.append(statescore)

    ##
    def backPropogation(self, stateList): #also modifies the global weight list
        errorList = []
        index = 0
        for state in stateList:
            actualScore =  self.examineGameState(state)
            neuralScore =  self.neuralScore(index)
            error = actualScore - neuralScore
            if(error > 1 or error < 0):
                print "Warning, invalid error at this node: " + str(error)
            self.deltaList.append(actualScore*(1 - actualScore)*error)
            self.errorList.append(weight[index]*deltaList[index])
            index += 1

        index = 0
        while index < self.neuralSize:
            currWeight = weightList[index]
            currDelta = self.deltaList[index]
            currNeurScore = self.neuralScore[index]
            weightList[index] = currWeight + alpha*currDelta*currNeurScore

   ##
    #depthSearch
    #Description: finds the best move in the given state, RECURSIVE
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #   originalState- The initial state of the game when the recursive function was called
    #   depth- The current depth of the search
    #
    #Return: The best move to be made if depth is 0, otherwise returns the best score seen
    # at the current depth
    ##
    def depthSearch(self, currentState, originalState, depth):
        movesList = listAllMovementMoves(currentState)
        movesList.append(Move(END, None, None))
        keys = ['Move', 'NextState', 'Score', 'Player']
        gameStateDic = []
        #get the current state score for comparison against child states
        currentStateScore = self.examineGameState(currentState)
        for move in movesList:
            nextState = getNextStateAdversarial(currentState, move)
            newStateScore = self.examineGameState(nextState)
            #if enemy turn, compare against the best seen enemy score (ab-prune)
            if (nextState.whoseTurn != originalState.whoseTurn
                and newStateScore > self.bestOverallScore):
                moveObject = [move, nextState, newStateScore, nextState.whoseTurn]
                gameStateDic.append(dict(zip(keys, moveObject)))
            #if its our turn, only add states to list with scores higher than the current state
            elif (newStateScore > currentStateScore):
                moveObject = [move, nextState, newStateScore, nextState.whoseTurn]
                gameStateDic.append(dict(zip(keys, moveObject)))

        #if haven't hit the limit, recurse on child states
        if depth != self.depthLimit:
            for state in gameStateDic:
                index = gameStateDic.index(state)
                if index < 6: #dont recurse on more than 5 nodes for speed
                    nextState = gameStateDic[index]['NextState']
                    nextPlayer = gameStateDic[index]['Player']
                    gameStateDic[index]['Score'] = self.depthSearch(nextState, originalState, depth+1)

        #base case, return the move that had the best score
        if depth == 0:
            bestMove = self.findBestMove(gameStateDic)
            if bestMove == None:
                return Move(END, None, None)
            else:
                self.stateList.append(getNextStateAdversarial(currentState, bestMove))
                self.bestOverallScore = 0
                return bestMove
        else: #bottom of tree, return the score
            bestScore = self.findBestScore(gameStateDic)
            if currentState.whoseTurn != originalState.whoseTurn:
                #see if there is a new best score to compare enemy nodes against
                if bestScore > self.bestOverallScore:
                    self.bestOverallScore = bestScore
                #as a simple food gatherer, enemy moves dont concern us outside pruning values
                bestScore = 0
            return bestScore

    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        return self.depthSearch(currentState, currentState, 0)

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #findBestMove
    #Looks at a list of nodes and determines the best average set of moves to make
    #
    #Parameters:
    #   dictList: List of dicts with the possible moves
    #
    #Return: The best move to be made

    def findBestMove(self, dictList):
        bestScore = -1000
        bestMove = None
        for item in dictList:
            if item['Score'] > bestScore:
                bestScore = item['Score']
                bestMove = item['Move']
        return bestMove
    ##
    #findBestScore
    #Looks at a list of nodes and determines the best score within the list
    #
    #Parameters:
    #   dictList: List of dicts with the possible moves
    #
    #Return: The best move to within the list

    def findBestScore(self, dictList):
        bestScore = -500
        for item in dictList:
            if item['Score'] > bestScore:
                bestScore = item['Score']
        return bestScore
  #registerWin
    #Is called when the game ends and simply indicates to the AI whether it has
    # won or lost the game.
    #
    #Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    #
    def registerWin(self, hasWon):
        index = 0
        for state in self.stateList:
            print "State Number: " + str(index)
            asciiPrintState(state)
        pass
