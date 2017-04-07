import random
import sys
import numpy as np
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

        for ant in myInventory.ants:
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
        utility = utility/2600
        return utility

    #Test method to create inputs using matrix and numpy library
    def generateInputs(self, currentState):

        #create empty matrix using numpy
        temp = np.empty([1, 12])

        #Variables
        myInven = None
        enemyInven = None
        enemy = None
        myInfo = None
        whichSide = 0
        workersArr = []
        numWorkers = 0

        if (currentState.inventory[0].player == currentState.whoseTurn):
            myInven = currentState.inventory[0]
            myInfo = myInv.player
            enemyInven = currentState.inventory[1]
            enemy = enemyInven.player
            whichSide = 1
        else:
            myInven = currentState.inventory[1]
            myInfo = myInven.player
            enemyInven = currentState.inventory[0]
            enemy = enemyInven.player
            whichSide = 0

        for ant in myInven.ants:
            if (ant.type == WORKER):
                workersArr.append(ant)
                numWorkers += 1


        if numWorkers >= 4:
            return -1

        #carrying food
        distanceValue = 0
        scale = 1
        foodLocation = getConstrList(currentState, None, (FOOD,))
        foodArr = []

        if(whichSide == 1):
            foodArr.append(foodLocation[2])
            foodArr.append(foodLocation[3])
        else:
            foodArr.append(foodLocation[0])
            foodArr.append(foodLocation[1])
        antHill = getConstrList(currentState, myInfo, (ANTHILL,))[0]
        tunnel = getConstrList(currentState, myInfo, (TUNNEL,))[0]
        for worker in workersArr:
            if (worker.carrying == TRUE):
                distanceValue += (scale / numWorkers) * 2.5
                if (approxDist(worker.coords, tunnel.coords) < approxDist(worker.coords, antHill,coords)):
                    distanceValue -= approxDist(worker.coords, tunnel.coords)
                else:
                    distanceValue -= approxDist(worker.coords, antHill.coords)
            else:
                distanceValue += (scale / numWorkers) * 2
    ##
    # g
    # Description: Calculate g(x)
    #
    #Parameters:
    #   x - value
    #
    ##
    def g(self, x):
        return 1/(1+math.exp(-x))


    ##
    # gPrime
    # Description: Calculate the transfer function g'(x)
    #
    #Parameter:
    #   x - value
    #
    # returns:
    #   the transfer derivative value at that point
    #
    def gPrime(self, x):
        return (x*(1.0 - x))

    ##
    # backPropogation
    # Description: modifies weight values after collecting an entire games worth
    #   of games states
    # Parameter:
    #   stateList - list of all states in a game, should be shuffled before inputs
    #
    def backPropogation(self, stateList): #also modifies the global weight list
        errorList = []

        #determines the error in each node
        index = 0
        for state in stateList:

            actualScore =  self.examineGameState(state)
            neuralScore =  self.neuralScoreEval(state)

            error = (actualScore - neuralScore)
            if(error > 1 or error < 0):
                print "Warning, invalid error at this node: " + str(error)
            deltaError = error*gPrime(actualScore) #amount of error in this node
            #self.deltaList.append(deltaError)

            errorList.append(weight[index]*deltaError)
            index += 1

        #adjusts/trains the weights
        index = 0
        for error in errorList:
            currWeight = weightList[index]
            currNeurScore = self.neuralScoreEval(stateList(index))
            currErr = errorList[index]
            weightList[index] = currWeight + alpha*currErr*currNeurScore
            index += 1

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
            nextState = getNextState(currentState, move)
            newStateScore = self.examineGameState(nextState)
            if (newStateScore > currentStateScore):
                moveObject = [move, nextState, newStateScore, nextState.whoseTurn]
                gameStateDic.append(dict(zip(keys, moveObject)))

        #if haven't hit the limit, recurse on child states
        if depth != self.depthLimit:
            for state in gameStateDic:
                index = gameStateDic.index(state)
                nextState = gameStateDic[index]['NextState']
                nextPlayer = gameStateDic[index]['Player']
                gameStateDic[index]['Score'] = self.depthSearch(nextState, originalState, depth+1)

        #base case, return the move that had the best score
        if depth == 0:
            bestMove = self.findBestMove(gameStateDic)
            if bestMove == None:
                self.stateList.append(currentState)
                return Move(END, None, None)
            else:
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
        if bestScore != -1000:
            print "Best Score: " + str(bestScore)
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
        #index = 0
        #for state in self.stateList:
        #    print "State Number: " + str(index)
        #    asciiPrintState(state)
        #    index ++
        #random.shuffle(self.stateList)
    #    self.neuralEvaluation(self.stateList)
    #    index = 0
    #    for score in self.neuralScore:
    #        print "neural score "+str(index)+" with value: " +str(score)
    #        index += 1
    #    self.backPropogation(self.stateList)
        pass
