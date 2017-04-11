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
        print "Starting New Game"
        super(AIPlayer,self).__init__(inputPlayerId, "Dumb Bunny")
        self.depthLimit = 1
        self.bestOverallScore = 0
        self.stateList = []

        #Variables needed for neural network
        self.alpha = .1
        self.sizeOfhiddenLayer = 5
        #Initialize matrices with random values
        self.firstWghtMatrix = np.matrix([[-0.12632104, -0.03244958,  0.58373195, -0.99145122,  0.58838048],
 [-0.41587605, -0.45706184,  1.09002623, -0.19254283,  1.15090878],
 [-0.21017106, -0.71604684, -0.83309405,  1.2280394,  -0.61419017],
 [ 0.19248349, -1.06138936, -0.89825464, -0.34204069,  0.2271802 ],
 [-0.35366484,  0.56528709, -0.51781602,  1.45648048,  0.0874081 ],
 [ 0.56064948, -0.37654417,  1.67631684, -1.0741196,   0.67191406],
 [-1.55634081,  0.65271412,  1.89875469, 0.80382882, -1.05273456],
 [-1.88695531, -2.19544389, -1.35166998,  2.13837717,  2.76567211],
 [-0.82753237,  1.68832913, -0.26015824, -1.7201558,   0.84773989],
 [ 0.44349653, -0.77205829, -0.89451787,  0.68695883, -1.32122501],
 [ 1.00476236, -0.00954299, -1.47968705, -0.24071018,  0.48171552]])

        self.secondWghtMatrix = np.matrix([[-2.75416984],
 [-2.05331264],
 [-1.02927015],
 [ 1.28730704],
 [ 2.00814793]])
        for i in range(0,11):
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
        if tunnel[0].coords:
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

    ############
    ##
    #generateInputs
    #
    #Description: Create a 1x11 matrix from the current state of the game
    # passed to the neural network
    #
    #Parameters:
    #   currentState - the state of the game to be evaluated.
    #
    #Return: an 1x11 matrix of
    #   higher values indicate moves that are closer to winning the game
    ##
    def generateInputs(self, currentState):

        #create empty matrix using numpy library
        matrix = np.empty([1, 11])

        #helpful variables used throughout method
        myInven = None
        enemyInven = None
        enemy = None
        myInfo = None
        whichSide = 0
        workersArr = []
        numWorkers = 0

        #check player inventories
        for inv in currentState.inventories:
            if inv.player == currentState.whoseTurn:
                myInven = inv
                whichSide = 1
            else:
                enemyInven = inv
                whichSide = 0
                for ant in enemyInven.ants:
                    if ant.type == QUEEN:
                        enemyQueenCoords = ant.coords
        for ant in myInven.ants:
            if (ant.type == WORKER):
                workersArr.append(ant)
                numWorkers += 1

        #food variables
        distanceValue = 0
        scale = 1
        foodLocation = getConstrList(currentState, None, (FOOD,))
        foodArr = []
        carryingWorkerVal = 0
        notCarryingWorkerVal = 0

        # check the distance of the worker value from the target goal
        if (whichSide == 1):
            foodArr.append(foodLocation[2])
            foodArr.append(foodLocation[3])
        else:
            foodArr.append(foodLocation[0])
            foodArr.append(foodLocation[1])
        antHill = getConstrList(currentState, myInfo, (ANTHILL,))[0]
        tunnel = getConstrList(currentState, myInfo, (TUNNEL,))[0]

        #check if the worker is carrying food or not and modify the values
        for worker in workersArr:
            if (worker.carrying == True):
                carryingWorkerVal += (scale / numWorkers)
                if (approxDist(worker.coords, tunnel.coords) < approxDist(worker.coords, antHill.coords)):
                    carryingWorkerVal -= approxDist(worker.coords, tunnel.coords) * 0.1
                else:
                    carryingWorkerVal -= approxDist(worker.coords, antHill.coords) * 0.1
            else:
                carryingWorkerVal += (scale / numWorkers) * 0.8
                if (approxDist(worker.coords, foodArr[0].coords) < approxDist(worker.coords, foodArr[1].coords)):
                    notCarryingWorkerVal -= approxDist(worker.coords, foodArr[0].coords)*0.1
                else:
                    notCarryingWorkerVal -= approxDist(worker.coords, foodArr[1].coords)*0.1

        #code to check the queen distance and change queen values
        distToQueen = 0
        for ant in myInven.ants:
            if ant.type != QUEEN and ant.type != WORKER and enemyInven.getQueen() != None:
                distToQueen += approxDist(ant.coords, enemyInven.getQueen().coords) * 0.1

        queenVal = 0.5
        #modify the queen's value based on anthill, tunnel, and food positions
        if approxDist(myInven.getQueen().coords, antHill.coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, tunnel.coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, foodArr[0].coords) < 2:
            queenVal -= 0.5
        if approxDist(myInven.getQueen().coords, foodArr[1]. coords) < 2:
            queenVal -= 0.5

        healthOfQueen = 0
        #modify the health of enemy queen value
        if enemyInven.getQueen() != None:
            healthOfQueen = float(enemyInven.getQueen().health) / 8.0
        #foodVale = my food count - enemy's food count raised to power of 1.5/30
        foodValue = math.pow(abs(myInven.foodCount - enemyInven.foodCount), 1.5) / 30.0
        #make food value negative is posittive and vice versa
        if enemyInven.foodCount > myInven.foodCount:
            foodValue *= -1

        # the inputs created that will go into the neural network
        matrix[0,0] = carryingWorkerVal
        matrix[0,1] = notCarryingWorkerVal
        matrix[0,2] = float(len(myInven.ants))/8.0
        matrix[0,3] = float(len(enemyInven.ants))/8.0
        matrix[0,4] = float(numWorkers)/float(len(myInven.ants))
        matrix[0,5] = float(myInven.getQueen().health)/8.0
        matrix[0,6] = healthOfQueen
        matrix[0,7] = foodValue
        matrix[0,8] = distToQueen
        matrix[0,9] = queenVal
        #Bias
        matrix[0,10] = 1

        return matrix #1x11 matrix of inputs

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

    ############
    ##
    #neuralNet
    #
    #Description: generates the output of the neural network given the
    # input array and using the current weights as well as implements a
    # backpropogation for adjusting the network’s weights using
    # gradient descent so that the agent can learn
    #
    #Parameters:
    #   inputs - 1x11 matrix of input values taken from generateInputs method
    #   targVal - the correct
    #
    #Return: the second layer output and error of output node
    #
    ##
    def neuralNet(self, inputs, targVal):

        #set first layer input to matrix multiplication of 2 arrays
        flInput = np.matmul(inputs, self.firstWghtMatrix) #1x5 matrix

        #create new 1x5 matrix
        flOutput = np.empty([1,self.sizeOfhiddenLayer])
        for i in range(0,self.sizeOfhiddenLayer):
            flOutput[0,i] = self.g(flInput[0,i])# Calculate g(x) for first layer

        #set second layer input to matrix multiplication of 2 arrays
        slInput = np.matmul(flOutput, self.secondWghtMatrix) # 1x1 matrix
        slOutput = self.g(slInput[0,0]) #calculates g(x) for 1x1 matrix

        #error = target - actual
        nodeOutputError = targVal - slOutput

        #change weights
        outputNodeDelta = nodeOutputError*(slOutput*(1-slOutput))#delta = error * g'(x)

        #calculate weights going into output node
        for i in range(0,self.sizeOfhiddenLayer):
            self.secondWghtMatrix[i,0] = self.secondWghtMatrix[i,0] + self.alpha*outputNodeDelta*flOutput[0,i]

        #calculate error of hidden layer nodes
        hiddenError = np.empty([1,self.sizeOfhiddenLayer]) #1x5 matrix
        for i in range(0,self.sizeOfhiddenLayer):
            hiddenError[0,i] = self.secondWghtMatrix[i,0]*outputNodeDelta
        #calculate delta of hidden layer perceptrons
        deltaHiddenLayer = np.empty([1,self.sizeOfhiddenLayer]) #1x5 matrix
        for i in range(0,self.sizeOfhiddenLayer):
            deltaHiddenLayer[0,i] = hiddenError[0,i]*flOutput[0,i]*(1-flOutput[0,i])

        #calculate weights of hidden layer inputs
        for i in range(0,11):
            for j in range(0,self.sizeOfhiddenLayer):
                self.firstWghtMatrix[i,j] = self.firstWghtMatrix[i,j] + self.alpha*deltaHiddenLayer[0,j]*inputs[0,i]
        #return second layer outputs and error of outputed node
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
            #newStateScore = self.examineGameState(nextState)

            matrix = self.generateInputs(nextState)
            #set first layer input to matrix multiplication of 2 arrays
            flInput = np.matmul(matrix, self.firstWghtMatrix) # 1x12*12x6 = 1x6 matrix

            #create new matrix array
            flOutput = np.empty([1,self.sizeOfhiddenLayer]) #1x6 matrix
            for i in range(0,self.sizeOfhiddenLayer):
                flOutput[0,i] = self.g(flInput[0,i])# Calculate g(x) for first layer

            #set second layer input to matrix multiplication of 2 arrays
            slInput = np.matmul(flOutput, self.secondWghtMatrix) # 1x6*6x1 = 1x1 matrix
            newStateScore = self.g(slInput[0,0]) #calculate g(x) for 1x1 matrix
            #[newStateScore, error] = self.neuralNet(matrix, self.examineGameState(nextState))
            if (newStateScore > currentStateScore):
                moveObject = [move, nextState, newStateScore, nextState.whoseTurn]
                gameStateDic.append(dict(zip(keys, moveObject)))
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
        bestScore = -2000
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
        bestScore = -3000
        for item in dictList:
            if item['Score'] > bestScore:
                bestScore = item['Score']
        return bestScore
    ###
    #registerWin
    #Is called when the game ends and simply indicates to the AI whether it has
    # won or lost the game. Also prints out the firstWghtMatrix and secondWghtMatrix
    # which are used to train the agent
    #
    #Parameters:
    #   hasWon - True if the player has won the game, False if the player lost. (Boolean)
    #
    def registerWin(self, hasWon):
        print " Game Over!: "
        print "     Modifying Weights"
        index = 0
        random.shuffle(self.stateList)
        while index < 1000: #need to see 1000 stable games before calling it okay
        #    error = self.backPropogation(self.stateList)
            for state in self.stateList:
                matrix = self.generateInputs(state)
                targetVal =  self.examineGameState(state)
                [output, error] = self.neuralNet(matrix, targetVal)
                if error < .03:
                    index += 1
                else:
                    index = 0
        print "Game Over"
        f = open('weights.txt', 'w')
        print >> f, self.firstWghtMatrix
        print >> f, self.secondWghtMatrix
        f.close()
        print "Output file done
        pass
