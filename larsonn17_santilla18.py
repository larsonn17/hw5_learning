import random
import sys
import numpy as np
import math
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

        #Variables needed for neural network
        self.alpha = .2
        self.sizeOfhiddenLayer = 6
        #Initialize matrices with random values
        self.firstWghtMatrix = np.empty([12, self.sizeOfhiddenLayer])
        for i in range(0,12):
            for j in range(0,self.sizeOfhiddenLayer):
                tempSize = random.uniform(-2,2)
                self.firstWghtMatrix[i,j] = tempSize

        self.secondWghtMatrix = np.empty([self.sizeOfhiddenLayer, 1])
        for i in range(0, self.sizeOfhiddenLayer):
            tempSize = random.uniform(-2, 2)
            self.secondWghtMatrix[i, 0] = tempSize
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

        utility = 100.0

        for inv in currentState.inventories:
            if inv.player == currentState.whoseTurn:
                myInventory = inv
            else:
                enemyInven = inv
                for ant in enemyInven.ants:
                    if ant.type == QUEEN:
                        enemyQueenCoords = ant.coords

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
                    #if food_1_distance < 1 or food_2_distance < 1:
                    #    utility += 100
                #if it does have food, have it find the closest drop off point
                if ant.carrying == True:
                    if (ant_to_tunnel < ant_to_anthill):
                        utility += 200 - (10 * ant_to_tunnel)
                    else:
                        utility += 200 - (10 * ant_to_anthill)
            else: #some other ant besides worker and queen
                utility += approxDist(ant.coords, enemyQueenCoords)

        if len(myInventory.ants) < 2:
            utility -= 200
        if len(enemyInven.ants) > 2:
            utility -= 200

        #score reduced by enemy queen health
        if enemyInven.getQueen() != None:
            utility -= enemyInven.getQueen().health * -10

        #score increased by amount of food
        utility += myInventory.foodCount*200
        #scale down to a value between 0 and 1
        utility = utility/2600
        return utility

    #Test method to create inputs using matrix and numpy library
    def generateInputs(self, currentState):

        #create empty matrix using numpy
        matrix = np.empty([1, 11])

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

        #carrying food
        distanceValue = 0
        scale = 1
        foodLocation = getConstrList(currentState, None, (FOOD,))
        foodArr = []
        carryingWorkerVal = 0
        notCarryingWorkerVal = 0

        if (whichSide == 1):
            foodArr.append(foodLocation[2])
            foodArr.append(foodLocation[3])
        else:
            foodArr.append(foodLocation[0])
            foodArr.append(foodLocation[1])

        antHill = getConstrList(currentState, myInfo, (ANTHILL,))[0]
        tunnel = getConstrList(currentState, myInfo, (TUNNEL,))[0]
        for worker in workersArr:
            if (worker.carrying == True):
                carryingWorkerVal += (scale / numWorkers)
                if (approxDist(worker.coords, tunnel.coords) < approxDist(worker.coords, antHill,coords)):
                    carryingWorkerVal -= approxDist(worker.coords, tunnel.coords) * 0.1
                else:
                    carryingWorkerVal -= approxDist(worker.coords, antHill.coords) * 0.1
            else:
                carryingWorkerVal += (scale / numWorkers) * 0.8
                if (approxDist(worker.coords, foodArr[0].coords) < approxDist(worker.coords, foodArr[1].coords)):
                    notCarryingWorkerVal -= approxDist(worker.coords, foodArr[0].coords)*0.1
                else:
                    notCarryingWorkerVal -= approxDist(worker.coords, foodArr[1].coords)*0.1
        distToQueen = 0
        for ant in myInven.ants:
            if ant.type != QUEEN and ant.type != WORKER and enemyInven.getQueen() != None:
                distToQueen += approxDist(ant.coords, enemyInven.getQueen().coords) * 0.1

        queenVal = 0.5

        if approxDist(myInven.getQueen().coords, antHill.coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, tunnel.coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, foodArr[0].coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, foodArr[1]. coords) < 2:
            queenVal -= 0.5

        healthOfQueen = 0
        if enemyInven.getQueen() != None:
            healthOfQueen = float(enemyInven.getQueen().health) / 8.0

        foodValue = math.pow(abs(myInven.foodCount - enemyInven.foodCount), 1.5) / 30.0
        if enemyInven.foodCount > myInven.foodCount:
            foodValue *= -1

        matrix[0,0] = carryingWorkerVal
        matrix[0,1] = notCarryingWorkerVal
        matrix[0,2] = float(len(myInven.ants))/8.0
        matrix[0,3] = float(len(enemyInven.ants))/8.0
        matrix[0,4] = float(numWorkers)/float(len(myInven.ants))
        matrix[0,5] = float(myInven.getQueen.health)/8.0
        matrix[0,6] = healthOfQueen
        matrix[0,7] = foodValue
        matrix[0,8] = distToQueen
        matrix[0,9] = queenVal
        #Bias
        matrix[0,10] = 1

        return matrix

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

    def neuralNet(self, inputs, targVal):

        #set first layer input to matrix multiplication of 2 arrays
        flInput = np.matmul(inputVars,self.firstWgtMatrix) # 1x12*12x6 = 1x6 matrix

        #create new matrix array
        flInput = np.empty([1,self.sizeOfhiddenLayer]) #1x6 matrix
        for i in range(0,self.sizeOfhiddenLayer):
            flInput[0,i] = self.g(flInput[0,i])# Calculate g(x) for first layer

        #set second layer input to matrix multiplication of 2 arrays
        slInput = np.matmul(flOutput, self.secondWghtMatrix) # 1x6*6x1 = 1x1 matrix
        slOutput = self.g(slInput[0,0]) #calculate g(x) for 1x1 matrix

        nodeOutputError = targVal - slOutput #error = target - actual

        #changing weights
        outputNodeDelta = nodeOutputError*(slOutput*(1-slOutput))#delta = Err * g'(x)

        #Calculate new Weights going into output node
        for i in range(0,self.sizeOfhiddenLayer):
            self.secondWghtMatrix[i,0] = self.secondWghtMatrix[i,0] + self.alpha*outputNodeDelta*flOutput[0,i]

        #Calculate error of Hidden Layer Nodes
        hiddenError = np.empty([1,self.sizeOfhiddenLayer]) # 1x6 matrix
        for i in range(0,self.sizeOfhiddenLayer):
            hiddenError[0,i] = self.secondWghtMatrix[i,0]*outputNodeDelta
        #Calculate delta of hidden layer perceptrons
        deltaHiddenLayer = np.empty([1,self.sizeOfhiddenLayer]) # 1x6 matrix
        for i in range(0,self.sizeOfhiddenLayer):
            deltaHiddenLayer[0,i] = hiddenError[0,i]*flOutput[0,i]*(1-flOutput[0,i])

        #Calculate New Weights of hidden layer inputs
        for i in range(0,12):
            for j in range(0,self.sizeOfhiddenLayer):
                self.firstWghtMatrix[i,j] = self.firstWghtMatrix[i,j] + self.alpha*deltaHiddenLayer[0,j]*inputs[0,i]


        return (slOutput, nodeOutputError)

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
                return Move(END, None, None)
            else:
                self.stateList.append(getNextState(currentState, bestMove))
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
        #if bestScore != -1000:
        #    print "Best Score: " + str(bestScore)
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
        bestScore = -1500
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
        random.shuffle(self.stateList)
        while index < 1000: #need to see 1000 stable games before calling it okay
        #    error = self.backPropogation(self.stateList)
            for state in stateList:
                matrix = self.generateInputs(state)
                targetVal =  self.examineGameState(state)
                [ouptut, error] = self.neuralNet(matrix, targetVal)
                if error < .05:
                    index += 1
                else:
                    index = 0
            print(np.matrix(self.firstWghtMatrix))
        pass
