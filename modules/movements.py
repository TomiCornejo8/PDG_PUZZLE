import numpy as np
import random

# Moves
RIGHT = (0,1)
LEFT = (0,-1)
UP = (-1,0)
DOWN = (1,0)
STAY = (0,0)

# Functions
def setRandomPlayer(map):
    voidSpace = np.argwhere(map == 0)
    player = voidSpace[np.random.choice(len(voidSpace))]
    return player

def lookAhead(player,move):
    return [player[0]+move[0],player[1]+move[1]]

def iceSliding(map,player,move):
    while map[lookAhead(player,move)[0],lookAhead(player,move)[1]] == 0:
        map[player[0],player[1]] = 6
        player = lookAhead(player,move)
    return player

def killEnemy(map,player,move):
    enemy = lookAhead(player,move)
    map[enemy[0],enemy[1]] = 0
    return map

def getEmpityTiles(map,player,move):
    tile = player.copy()
    voidArray = []
    notAllow = [1,2]
    while not map[lookAhead(tile,move)[0],lookAhead(tile,move)[1]] in notAllow:
        tile = lookAhead(tile,move)
    if map[tile[0],tile[1]] == 0: voidArray.append(tile.copy())
    return voidArray

def checkEdges(map,player):
    moves = [RIGHT, LEFT, UP, DOWN]
    for move in moves:
        tile = lookAhead(player,move)
        if map[tile[0],tile[1]] == 1: return move
    return STAY

def getAllowMoves(map,player):
    moves = [RIGHT, LEFT, UP, DOWN]
    notAllow = [1,2]
    newMoves = []
    for move in moves:
        tile = lookAhead(player,move)
        if not map[tile[0],tile[1]] in notAllow:
            newMoves.append(move)
    return newMoves
"""
def removeOppositeMove(moves,move):
    if move == LEFT:
        if RIGHT in moves: moves.remove(RIGHT)
    elif move == RIGHT:
        if LEFT in moves: moves.remove(LEFT)
    elif move == UP:
        if  DOWN in moves: moves.remove(DOWN)
    elif move == DOWN:
        if UP in moves: moves.remove(UP)
    return moves
"""

def getMap(map,maxMoves,blockFactor = 0.05,enemyFactor = 0.05):
    player = initialPlayer = setRandomPlayer(map)    
    currentMove = STAY
    moves = [RIGHT, LEFT, UP, DOWN]

    enemys = []

    contMoves = 0
    print(f"MAPA INICIAL\n{map}\n{player}\n")
    while contMoves < maxMoves or checkEdges(map,player) == STAY:    
        moves = getAllowMoves(map,player)
        print(f"ite {contMoves}\n moves {moves} player {player}\n {map}")
        if len(moves) == 0: break
        currentMove = random.choice(moves)
         
        r1 = random.uniform(0,1)
        if r1 <= blockFactor:
            voidArray = getEmpityTiles(map,player,currentMove)
            if len(voidArray) != 0: 
                block = random.choice(voidArray)
                map[block[0],block[1]] = 2
        
        player = iceSliding(map,player,currentMove)
        contMoves += 1 
        if contMoves == maxMoves: break
    
    move = checkEdges(map,player)
    door = lookAhead(player,move)
    map[door[0],door[1]] = 4
    map[initialPlayer[0],initialPlayer[1]] = 5
    
    return map,player